# Libraries Imported
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import argparse
#assert(torch.__version__ <= '1.1.0')
from model_utils import *
from data_utils import *
from datetime import datetime
from VisionTransformer import *
from datetime import datetime
from swin_vit import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Adding Parser
parser = argparse.ArgumentParser(description='BNN learning several tasks in a row, metaplasticity is controlled by the argument meta.')
#and command line arguments
parser.add_argument('--scenario', type = str, default = 'task', metavar = 'SC', help='1 mean per task or 1 mean for all task')
parser.add_argument('--net', type = str, default = 'bnn', metavar = 'NT', help='Type of net')
parser.add_argument('--in-size', type = int, default = 784, metavar = 'in', help='input size')
parser.add_argument('--hidden-layers', nargs = '+', type = int, default = [], metavar = 'HL', help='size of the hidden layers')
parser.add_argument('--out-size', type = int, default = 10, metavar = 'out', help='output size')
parser.add_argument('--task-sequence', nargs = '+', type = str, default = ['MNIST'], metavar = 'TS', help='Sequence of tasks to learn')
parser.add_argument('--lr', type = float, default = 0.001, metavar = 'LR', help='Learning rate')
parser.add_argument('--gamma', type = float, default = 1.0, metavar = 'G', help='dividing factor for lr decay')
parser.add_argument('--epochs-per-task', type = int, default = 5, metavar = 'EPT', help='Number of epochs per tasks')
parser.add_argument('--norm', type = str, default = 'bn', metavar = 'Nrm', help='Normalization procedure')
parser.add_argument('--meta', type = float, nargs = '+',  default = [0.0], metavar = 'M', help='Metaplasticity coefficients layer wise')
parser.add_argument('--rnd-consolidation', default = False, action = 'store_true', help='use shuffled Elastic Weight Consolidation')
parser.add_argument('--ewc-lambda', type = float, default = 0.0, metavar = 'Lbd', help='EWC coefficient')
parser.add_argument('--ewc', default = False, action = 'store_true', help='use Elastic Weight Consolidation')
parser.add_argument('--si-lambda', type = float, default = 0.0, metavar = 'Lbd', help='SI coefficient')
parser.add_argument('--si', default = False, action = 'store_true', help='use Synaptic Intelligence (SI)')
parser.add_argument('--bin-path', default = False, action = 'store_true', help='path integral on binary weight, else perform path integral on hidden weight')
parser.add_argument('--decay', type = float, default = 0.0, metavar = 'dc', help='Weight decay')
parser.add_argument('--init', type = str, default = 'uniform', metavar = 'INIT', help='Weight initialisation')
parser.add_argument('--init-width', type = float, default = 0.1, metavar = 'W', help='Weight initialisation width')
parser.add_argument('--save', type = bool, default = True, metavar = 'S', help='Saving the results')
parser.add_argument('--interleaved', default = False, action = 'store_true', help='saving results')
parser.add_argument('--beaker', default = False, action = 'store_true', help='use beaker')
parser.add_argument('--fb', type = float, default = 5e-3, metavar = 'fb', help='feeback coeff from last beaker to the first')
parser.add_argument('--n-bk', type = int, default = 4, metavar = 'bk', help='number of beakers')
parser.add_argument('--ratios', nargs = '+', type = float, default = [1e-2,1e-3,1e-4,1e-5], metavar = 'Ra', help='pipes specs between beakers')
parser.add_argument('--areas', nargs = '+', type = float, default = [1,2,4,8], metavar = 'Ar', help='beakers cross areas')
parser.add_argument('--device', type = str, default = 'cuda', metavar = 'Dev', help='choice of cpu or gpu')
parser.add_argument('--seed', type = int, default = None, metavar = 'seed', help='seed for reproductibility')
args = parser.parse_args()

if args.seed is not None:
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)


date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time+'_gpu'+str(args.device)
if not(os.path.exists(path)):
	os.makedirs(path)

createHyperparametersFile(path, args)


train_loader_list,test_loader_list, task_names= create_tiny_imgnet_loaders(root_dir='tiny-imagenet-200', num_subsets=5,batch_size=20, transform=transform)
# Checking the sizes of the created loaders
for i, (train_loader, test_loader) in enumerate(zip(train_loader_list, test_loader_list)):
    print(f"Subset {i + 1} - Train loader size: {len(train_loader.dataset)}, Test loader size: {len(test_loader.dataset)}")

# Hyperparameters
lr = args.lr
epochs = args.epochs_per_task
save_result = args.save
#meta = args.meta
ewc_lambda = args.ewc_lambda
si_lambda = args.si_lambda
archi = [args.in_size] + args.hidden_layers + [args.out_size]
# Assuming all necessary imports are done

# Define transformations (already provided by you)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Initialize the model based on the selected architecture
if args.net == 'vit':
    custom_config = {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "n_classes": 200,
        "embed_dim": 768,
        "depth": 4,
        "n_heads": 12,
        "mlp_ratio": 4.,
        "qkv_bias": True,
    }
    model = VisionTransformer(**custom_config).to(args.device)
elif args.net == 'bvit':
    custom_config = {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "n_classes": 200,
        "embed_dim": 768,
        "depth": 4,
        "n_heads": 12,
        "mlp_ratio": 4.,
        "qkv_bias": True,
    }
    model = BinarizeVisionTransformer(**custom_config).to(args.device)
elif args.net == 'hybrid':
    custom_config = {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "n_classes": 200,
        "embed_dim": 768,
        "depth": 4,
        "n_heads": 12,
        "mlp_ratio": 4.,
        "qkv_bias": True,
    }
    model = HybridisionTransformer(**custom_config).to(args.device)

# Initialize meta parameters
meta = {}
if args.net in ['vit', 'bvit', 'hybrid']:
    for n, p in model.named_parameters():
        index = [768, 10]
        p.newname = 'l' + str(index)
        if 'fc' in n or 'cv' in n:
            meta[p.newname] = args.meta[index - 1] if len(args.meta) > 1 else args.meta[0]

# Run the training loop
data = run_training_loop(model, train_loader_list, test_loader_list, args, epochs, task_names)

print(data)
