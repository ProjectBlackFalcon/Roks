import logging
import os
from tqdm import tqdm, trange
import numpy as np
import torch
from DataGenerator import get_data_loaders
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(model, eval_data_loader, device, training_loss, nb_training_steps, output_directory="runs"):
    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(eval_data_loader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels = batch
        with torch.no_grad():
            _, mc_loss = model(input_ids, labels=labels)

        eval_loss += mc_loss.mean().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    train_loss = training_loss / nb_training_steps
    result = {'eval_loss': eval_loss,
              'train_loss': train_loss}

    output_eval_file = os.path.join(output_directory, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def main():
    output_directory = "dofus"
    num_train_epochs = 1
    train_batch_size = 8
    eval_batch_size = 16
    learning_rate = 6.25-5
    weight_decay = 0.01

    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    train_data_loader, eval_data_loader = get_data_loaders(tokenizer, train_batch_size, eval_batch_size)

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
    for _ in trange(num_train_epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_data_loader, desc="Training")

        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, labels = batch
            losses = model(input_ids, labels=labels)
            loss = losses[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.defaults["lr"])

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_directory, WEIGHTS_NAME)
    output_config_file = os.path.join(output_directory, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_directory)

    # Load a trained model and vocabulary that you have fine-tuned
    model = GPT2LMHeadModel.from_pretrained(output_directory)
    model.to(device)

    # Evaluating
    evaluate(model, eval_data_loader, device, tr_loss, nb_tr_steps, output_directory)


if __name__ == '__main__':
    main()
