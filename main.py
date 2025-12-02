import json
from utils.data_util import prep_dataset
from utils.eval_util import evaluate_lora
from utils.input_parser import get_args
from utils.train_util import train_model

if __name__ == '__main__':
    print("Execution started")
    args = get_args()
    print(json.dumps(vars(args), indent=4))
    if args.do_snr:
        print("Dataset snr preparation")
        prep_dataset(args)
    if args.do_noisify:
        print("Dataset noise preparation")
        prep_dataset(args)
    if args.do_train:
        print("Training")
        train_model(args)
    if args.do_eval:
        print("Evaluation")
        evaluate_lora(args)
