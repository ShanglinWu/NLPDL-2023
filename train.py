from dataHelper import get_dataset
import datasets
import logging
import torch
import os
import random
from dataclasses import dataclass, field
import numpy as np
import evaluate
import transformers
import wandb
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

torch.cuda.set_per_process_memory_fraction(1.0, 0)
torch.cuda.empty_cache()


'''
settings
'''
lr = 1e-5
epoch_num = 5
batch_size = 16


classes_dic = {'restaurant_sup': 3, 'acl_sup': 6, 'agnews_sup': 4}



'''
Initialize metircs
'''
from datasets import load_metric
metric = load_metric("./metric.py")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result

@dataclass
class Modelargs():
    model_name: str
    add_adaptor: bool

@dataclass
class Dataargs():
    dataset_name: str


logger = logging.getLogger(__name__)


def main():


        os.environ["HF_EVALUATE_OFFLINE"] = "1"

        '''
        initialize logging, seed, argparse...
        '''

        parser = HfArgumentParser((Modelargs, Dataargs, TrainingArguments))
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

        dataset_name = dataset_args.dataset_name
        model_name = model_args.model_name

        print("-----------------------------------------------------------------")
        print("Now we are running:")
        print("Dataset: "+ dataset_name)
        print("Model: " + model_name)
        print("Adaptor: " + str(model_args.add_adaptor))
        print("-----------------------------------------------------------------")

        training_args = TrainingArguments(
            output_dir=".\\results",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch_num,
            logging_dir='./logs',            # directory for storing logs
            logging_steps=25,
            evaluation_strategy='steps',
            save_strategy='epoch',
            report_to='wandb',
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        
        


        set_seed(training_args.seed)


        '''
            load datasets
        '''

        dataset = get_dataset(dataset_name, "<SEP>")
        
        
        '''
            load models
        '''

        config = AutoConfig.from_pretrained(model_name, num_labels=classes_dic[dataset_name])
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="SEQ_CLS",
        )


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
        if model_args.add_adaptor:
            model =  get_peft_model(model, peft_config)






        '''
            process datasets and build up datacollator
        '''

        def tokenize_dataset(dataset):
            return tokenizer(dataset["text"],truncation=True, padding=True, max_length=16, return_tensors="pt")
        
        dataset = dataset.map(tokenize_dataset, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            # compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        wandb.init(project='Adaptor_v1.0', name=dataset_name+'-'+model_name, config=training_args)
        wandb.watch(model)
        # make the wandb obeserver to log the accuracy, micro_f1, macro_f1
        # wandb.log({'accuracy': trainer.compute_metrics['accuracy']})
        # define the name of the runs\


        trainer.train()

        trainer.evaluate()
        wandb.finish()
        


    

    

        
    




if __name__ == "__main__":
    main()
