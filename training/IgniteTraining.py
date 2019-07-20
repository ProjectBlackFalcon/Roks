import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from pprint import pformat
from DataGenerator import get_data_loaders
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, OpenAIAdam, CONFIG_NAME
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, MetricsLambda
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from tqdm import tqdm, trange

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits[0]
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k and top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    logits.unsqueeze_(0)
    return logits


def sample_sequence(model, tokenizer, device, length=10, context=None, temperature=1.0):
    if context is not None:
        context = tokenizer.encode(context)
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
    else:
        context = torch.tensor([tokenizer.encoder["<|endoftext|>"]], device=device, dtype=torch.long).unsqueeze(0)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for _ in trange(length):
            logits, past = model(prev, past=past)
            logits = top_k_top_p_filtering(logits[:, -1, :] / temperature, top_p=0.9, top_k=40)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
            output = torch.cat((output, prev), dim=1)

    output = output[0].tolist()
    output = tokenizer.decode(list(filter(lambda token: token != 50256, output)))
    return output


def run(train_batch_size, val_batch_size, epochs, log_interval, lr):
    train_loader, val_loader = get_data_loaders(tokenizer, train_batch_size, val_batch_size)

    optimizer = OpenAIAdam(model.parameters(), lr=6.25e-5)

    def update(engine, batch):
        model.train()
        batch = batch[0].to(device)
        loss = model.forward(batch, lm_labels=batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def inference(engine, batch):
        print(sample_sequence(model, length=20))

        model.eval()
        with torch.no_grad():
            x, y = batch
            logits, _ = model(x)
            logits = logits[:, -1, :]
            logits = torch.softmax(logits, dim=-1)
        return logits, y

    metrics = {'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)}
    metrics['ppl'] = MetricsLambda(math.exp, metrics['nll'])

    trainer = Engine(update)
    evaluator = Engine(inference)

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, lr), (epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        avg_ppl = metrics["ppl"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Ppl: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll, avg_ppl)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        avg_ppl = metrics['ppl']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Ppl: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll, avg_ppl))

        pbar.n = pbar.last_print_n = 0

    tb_logger = TensorboardLogger(None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)

    @evaluator.on(Events.COMPLETED)
    def tb_log_metrics(engine):
        for name in metrics.keys():
            tb_logger.writer.add_scalar(name, engine.state.metrics[name], trainer.state.iteration)

    checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
        'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
    tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    print(sample_sequence(model, length=20))
    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=6.25e-5,
                        help='learning rate (default: 6.25e-5)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how ma'
                             'ny batches to wait before logging training status')

    args = parser.parse_args()

    # sample_sequence(model, length=10)
    run(args.batch_size, args.val_batch_size, args.epochs, args.log_interval, args.lr)

