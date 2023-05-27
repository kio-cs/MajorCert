import torch
import torch.backends.cudnn as cudnn
import pytorch_cifar.models.resnet as resnet

import torchvision
import torchvision.transforms as transforms
import argparse
from utils_for_certify_cifar import generate_smoothing_units, \
    generate_prediction_for_units, static_certify,turn_int_to_tensor,quality_of_output_wrongcert, \
    result_shower_incorrect_cert, dynamic_single_certify_forthe4_fast, quality_of_output_wrongcert_theo
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR Certification')

# parameter setting
parser.add_argument('--block_size', default=12, type=int, help='size of each smoothing band')
parser.add_argument('--row_size', default=4, type=int, help='size of each smoothing band')
parser.add_argument('--band_size', default=4, type=int, help='size of each smoothing band')
parser.add_argument('--size_to_certify', default=5, type=int, help='size_to_certify')
parser.add_argument('--threshhold', default=0.3, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--static_certify', default=True, type=bool, help='whether use static certify')
parser.add_argument('--the3_based_on_dyanmic_fast', default=True, type=bool, help='whether use static certify')

sample_size=32
args = parser.parse_args()
print("args.size_to_certify",args.size_to_certify)
torch.set_printoptions(profile="full")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
band_size_dict = {"block": args.block_size, "column": args.band_size, "row": args.row_size}
# set the dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# load model
print('==> Building model..')
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

print('==> Resuming from checkpoint..')
# write down address
model_address = {
    "block": 'block.pth',
    "column": 'column.pth',
    "row": 'row.pth'
}
print(model_address)
model_dict = {}
normalize_weight_dict = {"block": 1 / 32, "column": 1, "row": 1}

# load from address
for model_name in model_address:
    if (args.model == 'resnet50'):
        net = resnet.ResNet50()
    elif (args.model == 'resnet18'):
        net = resnet.ResNet18()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)
    resume_file = '../{}/{}'.format(checkpoint_dir, model_address.get(model_name))
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)

    net.load_state_dict(checkpoint['net'])
    net.eval()
    model_dict[model_name] = net


# the main entrance of certify
def main_certify_cal():
    static_clean_correct_dict = {}
    static_cert_correct_dict = {}
    static_cert_incorrect_dict = {}

    the3_dynamic_fast_clean_correct_dict = {}
    the3_dynamic_fast_cert_correct_dict = {}
    the3_dynamic_fast_cert_incorrect_dict = {}

    the1_cert_correct=0
    the2_cert_correct = 0

    the1_cert_incorrect=0
    the2_cert_incorrect = 0


    total = 0
    start_time = time.time()


    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += len(inputs)
        # generate_smoothing_units here
        units_dict = generate_smoothing_units(inputs, band_size_dict)
        # put them into model
        with torch.no_grad():
            softmx_dict, vote_sum_dict, vote_each_list_dict = generate_prediction_for_units(inputs=inputs,
                                                                                            units_dict=units_dict,
                                                                                            model_dict=model_dict,
                                                                                            num_classes=10,
                                                                                            threshold=args.threshhold)
        # ready, start certify
        # static or not? static means DRS
        if args.static_certify:
            for smoothing_method in model_dict:
                # certify
                idx_first, idx_second, val_first, val_second, cert = static_certify(vote_sum_dict.get(smoothing_method),
                                                                                    args.size_to_certify,
                                                                                    smoothing_method, band_size_dict)
                # see see whether is correct
                static_clean_correct_dict, static_cert_correct_dict,static_cert_incorrect_dict = quality_of_output_wrongcert(idx_first, targets, cert,
                                                                                        smoothing_method,
                                                                                        static_clean_correct_dict,
                                                                                        static_cert_correct_dict,static_cert_incorrect_dict)

            result_shower_incorrect_cert(static_clean_correct_dict, static_cert_correct_dict,static_cert_incorrect_dict, total, time.time() - start_time,
                          "static")

        if args.the3_based_on_dyanmic_fast:
            prediction_output, cert_output,theo=dynamic_single_certify_forthe4_fast(softmx_dict, args.size_to_certify, band_size_dict, args.threshhold)
            prediction_output = turn_int_to_tensor(prediction_output)
            the3_dynamic_fast_clean_correct_dict, the3_dynamic_fast_cert_correct_dict, the3_dynamic_fast_cert_incorrect_dict,the1_cert_correct,the2_cert_correct,the1_cert_incorrect,the2_cert_incorrect = quality_of_output_wrongcert_theo(prediction_output,
                                                                                                targets, cert_output,
                                                                                                "the3_based_on_dyanmic_fast",
                                                                                                the3_dynamic_fast_clean_correct_dict,
                                                                                                the3_dynamic_fast_cert_correct_dict,
                                                                                                          the3_dynamic_fast_cert_incorrect_dict,the1_cert_correct,the2_cert_correct,the1_cert_incorrect,the2_cert_incorrect,theo)
            # show the result
            result_shower_incorrect_cert(the3_dynamic_fast_clean_correct_dict, the3_dynamic_fast_cert_correct_dict, the3_dynamic_fast_cert_incorrect_dict, total,
                          time.time() - start_time,
                          "the3_based_on_dyanmic_fast")
            print("the1_cert_correct",the1_cert_correct)
            print("the2_cert_correct",the2_cert_correct)
            print("the1_cert_incorrect",the1_cert_incorrect)
            print("the2_cert_incorrect",the2_cert_incorrect)

main_certify_cal()
