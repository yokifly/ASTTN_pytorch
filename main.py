import time
import argparse
import torch.optim as optim
from utils import log_string, masked_mae
from utils import count_parameters, load_data, load_graph
# from model.model_both import Model_Both

from model.train import train
from model.model_adp import Model_Adp
from model.model_both import Model_Both


parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of block layers')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--window', type=int, default=6,
                    help='temporal window size for attentions')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=20,
                    help='decay epoch')
parser.add_argument('--ds', default='pems-bay',
                    help='dataset name')
parser.add_argument('--se_type', default='node2vec',
                    help='spatial embedding file')
parser.add_argument('--remark', default='',
                    help='remark')
parser.add_argument('--model', default='adp',
                    help='model_type')

args = parser.parse_args()

args.traffic_file = './data/' + args.ds + '.h5'
args.model_file = './save/' + args.ds + '_' + args.model + args.remark + '.pkl'

save_log =  './save/'+args.ds+'_'+args.model+args.remark + '.log'
log = open(save_log, 'a')
log_string(log, "New...")
log_string(log, str(args)[10: -1])

T = 24 * 60 // args.time_slot

# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, mean, std) = load_data(args)
SE, g = load_graph(args)

num_nodes = trainY.shape[2]
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std
# build model
log_string(log, 'compiling model...')

if args.model == "adp":
    model = Model_Adp(SE, args, N = num_nodes, T = args.num_his, window_size = args.window)
elif args.model == "both":
    model = Model_Both(SE, args, N = num_nodes, T=args.num_his, window_size=args.window, g = g)
    
loss_criterion = masked_mae
optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)
parameters = count_parameters(model)

log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)