#Version 1.0 for proposed design.
#This this version, prefetch optimization includes two stages and compare with the pytorch loading
#No training process is needed in this version since the optimization is on data loading I/O
#In the next version will include training on PtychoNN for the third stage.

from __future__ import print_function
import os
import argparse
import time
import socket
import random
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle
#from skimage.transform import resize
import skimage.transform
import multiprocessing as mp
from ctypes import *
#MPI setting

def get_local_rank(required=False):
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    if required:
        raise RuntimeError('Could not get local rank')
    return 0


def get_local_size(required=False):
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    if 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    if required:
        raise RuntimeError('Could not get local size')
    return 1


def get_world_rank(required=False):
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    if required:
        raise RuntimeError('Could not get world rank')
    return 0


def get_world_size(required=False):
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    if required:
        raise RuntimeError('Could not get world size')
    return 1


# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    local_rank=get_local_rank()
    rank=get_world_rank()
    size=get_world_size()

    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)

    if local_rank == 0:
        print("This is GPU 0 from node: %s" %(socket.gethostname()))

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)

#Parse input parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)
parser.add_argument('--model_save_path', required=True, help='path to save model', type=str)
args = parser.parse_args()

#DDP backend setting
# What backend?  nccl on GPU, gloo on CPU
if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

if with_ddp:
    torch.distributed.init_process_group(
        backend=backend, init_method='env://')


torch.manual_seed(args.seed)

if args.device == 'gpu':
    # DDP: pin GPU to local rank.
    torch.cuda.set_device(int(local_rank))
    torch.cuda.manual_seed(args.seed)

if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

if rank==0:
    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())

path = os.getcwd()

MODEL_SAVE_PATH =args.model_save_path
#if (not os.path.isdir(MODEL_SAVE_PATH)):
    #os.mkdir(MODEL_SAVE_PATH)

nepochs=args.epochs
LOCAL_BATCH_SIZE = args.batch_size
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE*size
# define parameters
NGPUS = size
BATCH_SIZE = GLOBAL_BATCH_SIZE
#LR = NGPUS * args.lr
LR = args.lr
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)

# x,y dimension size
H, W = 64, 64

# training, test, validation data size
NLINES = 100  #How many lines of data to use for training?
NLTEST = 60  #How many lines for the test set?

N_TRAIN = NLINES * 161
N_VALID = 805  #How much to reserve for validation


# load data
# data path
data_path = '/path/to/PtychoNN/data/20191008_39_diff.npz'
label_path = '/path/to/PtychoNN/data/20191008_39_amp_pha_10nm_full.npy'

data_diffr = np.load(data_path)['arr_0']
real_space = np.load(label_path)
amp = np.abs(real_space)
ph = np.angle(real_space)
print(amp.shape)
print(data_diffr.shape)

# crop diff to (64,64)
data_diffr_red = np.zeros(
    (data_diffr.shape[0], data_diffr.shape[1], 64, 64), float)
for i in tqdm(range(data_diffr.shape[0])):
    for j in range(data_diffr.shape[1]):
        data_diffr_red[i, j] = skimage.transform.resize(data_diffr[i, j, 32:-32, 32:-32],
                                        (64, 64),
                                        preserve_range=True,
                                        anti_aliasing=True)
        data_diffr_red[i, j] = np.where(data_diffr_red[i, j] < 3, 0,
                                        data_diffr_red[i, j])

# split training and testing data
tst_strt = amp.shape[0] - NLTEST  #Where to index from
print(tst_strt)

X_train = data_diffr_red[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]
Y_I_train = amp[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]
Y_phi_train = ph[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]

print(X_train.shape)

X_train, Y_I_train, Y_phi_train = shuffle(X_train,
                                            Y_I_train,
                                            Y_phi_train,
                                            random_state=0)

#Training data
X_train_tensor = torch.Tensor(X_train)
Y_I_train_tensor = torch.Tensor(Y_I_train)
Y_phi_train_tensor = torch.Tensor(Y_phi_train)

print(Y_phi_train.max(), Y_phi_train.min())

print(X_train_tensor.shape, Y_I_train_tensor.shape,
        Y_phi_train_tensor.shape)

train_data = TensorDataset(X_train_tensor, Y_I_train_tensor,
                            Y_phi_train_tensor)

# split training and validation data
train_data2, valid_data = torch.utils.data.random_split(
    train_data, [N_TRAIN - N_VALID, N_VALID])
print(len(train_data2), len(valid_data))  #, len(test_data)

#download and load training data
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data2, num_replicas=size, shuffle=True, rank=rank)
train_loader = torch.utils.data.DataLoader(train_data2,
                            batch_size=LOCAL_BATCH_SIZE,
                            sampler=train_sampler,
                            num_workers=4)

valid_sampler = torch.utils.data.distributed.DistributedSampler(
    valid_data, num_replicas=size, shuffle=True, rank=rank)
valid_loader = torch.utils.data.DataLoader(valid_data,
                            batch_size=LOCAL_BATCH_SIZE,
                            sampler=valid_sampler,
                            num_workers=4)

nsamples = len(train_data2)
# define network
nconv = 32

class recon_model(nn.Module):
    def __init__(self):
        super(recon_model, self).__init__()

        self.encoder = nn.Sequential(  # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            nn.Conv2d(in_channels=1,
                      out_channels=nconv,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(nconv, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(nconv * 2, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  #Amplitude model
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  #Phase model
        )

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph * np.pi  #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph

model = recon_model()


if args.device == 'gpu':
    # Move model to GPU.
    model.cuda()

if with_ddp:
    # wrap the model in DDP:
    model = DDP(model)
    

# Horovod: scale learning rate by the number of GPUs.

#Optimizer details
iterations_per_epoch = np.floor(nsamples/GLOBAL_BATCH_SIZE)+1 #Final batch will be less than batch size
step_size = 6*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
if local_rank==0:
    print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10, max_lr=LR, step_size_up=step_size,
                                              cycle_momentum=False, mode='triangular2')
#Function to update saved model if validation loss is minimum
def update_saved_model(model, model_path):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    #for f in os.listdir(model_path):
        #os.remove(os.path.join(model_path, f))
    if (NGPUS>1):    
        torch.save(model.module.state_dict(),model_path+'base_model_gpu'+str(NGPUS)+'.pth') #Have to save the underlying model else will always need 4 GPUs
    else:
        torch.save(model,model_path+'base_model_gpu'+str(NGPUS)+'.pth')

def metric_average(val, name):
    if (with_ddp):
        # Sum everything and divide by total size:
        dist.all_reduce(val,op=dist.ReduceOp.SUM)
        val /= size
    else:
        pass
    return val

""" Gradient averaging. """
def average_gradients(model,step,rank):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

#Profiling
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/path/to/save/trace/'),
        record_shapes=True,
        with_stack=True)
load_num=[]

def train(train_loader,metrics,epoch):
    total_io=0
    tot_loss = torch.tensor(0.0)
    loss_amp = torch.tensor(0.0)
    loss_ph = torch.tensor(0.0)
    if args.device == 'gpu':
        tot_loss = tot_loss.cuda()
        loss_amp = loss_amp.cuda()
        loss_ph = loss_ph.cuda()
    
    #data_times = AverageTracker()
    
    
    #total_time_arr=np.zeros([size,args.epochs])
    comp_time = 0
    step_time = 0
    grad_avg_time = 0
    start_time = time.perf_counter()
    for i, (ft_images,amps,phs) in tqdm(enumerate(train_loader)):
        if args.device == "gpu":
            ft_images = ft_images.cuda() #Move everything to device
            amps = amps.cuda()
            phs = phs.cuda()
        step_start = time.perf_counter()
        pred_amps, pred_phs = model(ft_images) #Forward pass
        #Compute losses
        loss_a = criterion(pred_amps,amps) #Monitor amplitude loss
        loss_p = criterion(pred_phs,phs) #Monitor phase loss but only within support (which may not be same as true amp)
        loss = loss_a + loss_p #Use equiweighted amps and phase

       #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss.backward()
        step_time = time.perf_counter() - step_start
        comp_time += step_time
        average_gradients(model,i,rank)
        optimizer.step()
        #profiler.step()
        #prof.step()
        tot_loss += loss.detach().item()
        loss_amp += loss_a.detach().item()
        loss_ph += loss_p.detach().item()
        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step()
        metrics['lrs'].append(scheduler.get_last_lr())
        
    end_time = time.perf_counter()
    total = end_time - start_time
    data_loading = total - comp_time
    metrics['compTime'].append(comp_time)
    metrics['loadTime'].append(data_loading)
    tot_loss_avg = metric_average(tot_loss, 'loss')
    loss_amp_avg = metric_average(loss_amp, 'loss')
    loss_ph_avg = metric_average(loss_ph, 'loss')
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss_avg/i,loss_amp_avg/i,loss_ph_avg/i]) 

def validate(validloader,metrics):
    tot_val_loss = torch.tensor(0.0)
    val_loss_amp = torch.tensor(0.0)
    val_loss_ph = torch.tensor(0.0)
    if args.device == 'gpu':
        tot_val_loss = tot_val_loss.cuda()
        val_loss_amp = val_loss_amp.cuda()
        val_loss_ph = val_loss_ph.cuda()
    for j, (ft_images,amps,phs) in enumerate(validloader):
        if args.device == "gpu":
            ft_images = ft_images.cuda()
            amps = amps.cuda()
            phs = phs.cuda()
        pred_amps, pred_phs = model(ft_images) #Forward pass
    
        val_loss_a = criterion(pred_amps,amps) 
        val_loss_p = criterion(pred_phs,phs)
        val_loss = val_loss_a + val_loss_p
    
        tot_val_loss += val_loss.detach().item()
        val_loss_amp += val_loss_a.detach().item()
        val_loss_ph += val_loss_p.detach().item()
    
    tot_val_loss_avg = metric_average(tot_val_loss, 'loss')
    val_loss_amp_avg = metric_average(val_loss_amp, 'loss')
    val_loss_ph_avg = metric_average(val_loss_ph, 'loss')
    metrics['val_losses'].append([tot_val_loss_avg/j,val_loss_amp_avg/j,val_loss_ph_avg/j])
  
  #Update saved model if val loss is lower
    if(tot_val_loss_avg/j<metrics['best_val_loss']) and rank==0:
        print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss_avg/j))
        metrics['best_val_loss'] = tot_val_loss_avg/j
        update_saved_model(model, MODEL_SAVE_PATH)


metrics = {'compTime':[], 'loadTime':[],'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
#metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
dur = []
loss=[]
loss_amp=[]
loss_ph=[]
val_loss=[]
val_loss_amp=[]
val_loss_phase=[]
training_start_time = time.time()
comp_time_arr=np.zeros([size,args.epochs])
load_time_arr=np.zeros([size,args.epochs])


for epoch in range (args.epochs):
    #train_sampler.set_epoch(epoch)
    train_loader.sampler.set_epoch(epoch)
    #Set model to train mode
    model.train()
    #Training loop
    t0 = time.time()
    #prof.start()
    train(train_loader,metrics,epoch)
    #prof.stop()
    dur.append(time.time() - t0)
    #Switch model to eval mode
    model.eval()
    #Validation loop
    validate(valid_loader,metrics)
    loss.append(metrics['losses'][-1][0].item())
    loss_amp.append(metrics['losses'][-1][1].item())
    loss_ph.append(metrics['losses'][-1][2].item())
    val_loss.append(metrics['val_losses'][-1][0].item())
    val_loss_amp.append(metrics['val_losses'][-1][1].item())
    val_loss_phase.append(metrics['val_losses'][-1][2].item())
    if rank==0:
        print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
        print('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
        print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
        print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))
total_training_time=time.time() - training_start_time
if rank==0:
    print("*******************************************")
    print("DDP? "+str(with_ddp))
    print("Number of GPUs used: "+str(NGPUS))
    print("Epoch total number: "+str(args.epochs))
    print("Local Batch Size: "+str(LOCAL_BATCH_SIZE)+" Learning Rate: "+str(LR))
    print("Minimum Validation Loss: "+str(np.amin(np.asarray(val_loss))))
    print("Total training time: {:.2f} seconds".format(total_training_time))
    print("Loss: "+str(loss))
    print("Loss amp: "+str(loss_amp))
    print("Loss phase: "+str(loss_ph))
    print("Validation Loss: "+str(val_loss))
    print("Validation Loss amp: "+str(val_loss_amp))
    print("Validation Loss phase: "+str(val_loss_phase))
    print("Time for each epoch: "+str(dur))
    print("Loading time for each rank each epoch: ")
    #print("*******************************************")
print("Loading time Rank %s: %s" %(rank,metrics['loadTime']))
print("Computation time Rank %s: %s" %(rank,metrics['compTime']))