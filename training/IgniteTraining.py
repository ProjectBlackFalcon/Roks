from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from DataGenerator import get_data_loaders

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, OpenAIAdam

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, MetricsLambda

from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def run(train_batch_size, val_batch_size, epochs, log_interval):
    train_loader, val_loader = get_data_loaders(tokenizer, train_batch_size, val_batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.log_interval)

