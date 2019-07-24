import logging
import os
from tqdm import tqdm, trange
import numpy as np
import torch
from DataGenerator import get_data_loaders
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME)
from torch.utils.tensorboard import SummaryWriter
from Utils import log_tensorboard

writer = SummaryWriter()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def save(model, tokenizer, output_directory, name=None):
    if name:
        output_directory = output_directory + "_" + name
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_directory, WEIGHTS_NAME)
    output_config_file = os.path.join(output_directory, CONFIG_NAME)

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_directory)


def evaluate(model, tokenizer, eval_data_loader, device, training_loss, previous_loss, nb_training_steps, output_directory="runs"):
    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch_element in tqdm(eval_data_loader, desc="Evaluating"):
        with torch.no_grad():
            mc_loss = model(batch_element, labels=batch_element)[0]

        eval_loss += mc_loss.mean().item()
        nb_eval_examples += batch_element.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    train_loss = training_loss / nb_training_steps
    result = {'eval_loss': eval_loss,
              'train_loss': train_loss}

    if eval_loss < previous_loss:
        print(eval_loss, "<", previous_loss, "saving best model.")
        save(model, tokenizer, output_directory, "best")

    output_eval_file = os.path.join(output_directory, "eval_results.txt")
    with open(output_eval_file, "w") as output_eval:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            output_eval.write("%s = %s\n" % (key, str(result[key])))

    return min(eval_loss, previous_loss)


def main():
    output_directory = "dofus"
    num_train_epochs = 3
    train_batch_size = 4
    eval_batch_size = 2
    max_context_length = 512
    learning_rate = 6.25e-5
    weight_decay = 0.01

    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    global_step = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    train_data_loader, eval_data_loader = get_data_loaders(tokenizer, train_batch_size, eval_batch_size, max_context_length=max_context_length, device=device)

    # Preparing the optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)

    # Training the model
    model.train()
    previous_loss = float("inf")

    for _ in trange(num_train_epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_data_loader, desc="Training")

        for step, batch_element in enumerate(tqdm_bar):
            logger.info(batch_element.size())
            losses = model(batch_element, labels=batch_element)
            loss = losses[0]

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            global_step += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.defaults["lr"])

            if step % 1000 == 0:
                save(model, tokenizer, output_directory)

            if step % 100 == 0:
                log_tensorboard(model, writer, global_step, exp_average_loss, tokenizer, device)

            optimizer.zero_grad()

        previous_loss = evaluate(model, tokenizer, eval_data_loader, device, tr_loss, previous_loss,
                                 nb_tr_steps, output_directory)
        model.train()

    save(model, tokenizer, output_directory)

    # Evaluating
    evaluate(model, eval_data_loader, device, tr_loss, nb_tr_steps, output_directory)


if __name__ == '__main__':
    main()
