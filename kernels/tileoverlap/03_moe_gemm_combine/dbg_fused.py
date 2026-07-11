import os,sys
sys.path.append("/data/cinnzhang_vllm_td_test/xxy/moe_bench/ThunderKittens/kernels/parallel")
import torch
from common import init_distributed_environment, destroy_distributed_environment, check_diff
from _Cf import TKParallelTensor, moe_gemm_combine_fused
ROW_BLOCK=128; TOP_K=8
lr,lws=init_distributed_environment()
device=f"cuda:{lr}"
S=2048;H=7168;I=2048;E=256;top_k=8
npd=S//lws; ne=E//lws
torch.random.manual_seed(1234)
rw=torch.rand(E,device=device); chosen=torch.multinomial(rw.repeat(S,1),top_k,replacement=False).to(torch.int32)
tpe=torch.bincount(chosen.view(-1),minlength=E).to(torch.int32)
pad=(tpe+127)//128*128
es=ne*lr; ee=ne*(lr+1)
npl=int(pad[es:ee].sum()); npm=int(pad.reshape(lws,ne).sum(1).amax())
torch.random.manual_seed(42+lr)
h=torch.randn(npl,I,device=device,dtype=torch.bfloat16)/I**0.5
w=torch.randn(ne,I,H,device=device,dtype=torch.bfloat16)/I**0.5
eo=TKParallelTensor((npm,H),dtype=torch.bfloat16,local_rank=lr,local_world_size=lws,multicast=False); eo.data_.zero_()
bar=TKParallelTensor((1+lws,max(npm//128+1,32)),dtype=torch.int,local_rank=lr,local_world_size=lws,multicast=False); bar.data_.zero_()
# combine idx: all -1 so combine does nothing; we only check the GEMM half
num_src=npd
cidx=torch.full((num_src*TOP_K,2),-1,dtype=torch.int32,device=device)
cw=torch.zeros((num_src*TOP_K,1),dtype=torch.float32,device=device)
cout=torch.zeros(num_src,H,device=device,dtype=torch.bfloat16)
torch.distributed.barrier(); torch.cuda.synchronize()
moe_gemm_combine_fused(h,w,eo,pad,cout,cidx,cw,bar,16,npl,num_src,1)
torch.cuda.synchronize()
# reference GEMM
eoref=torch.zeros(npl,H,device=device,dtype=torch.bfloat16)
st=0
for e in range(ne):
    en=st+int(pad[es+e])
    if en>st: torch.matmul(h[st:en],w[e],out=eoref[st:en])
    st=en
check_diff("expert_outputs GEMM half", eo.data_[:npl], eoref)
destroy_distributed_environment()
