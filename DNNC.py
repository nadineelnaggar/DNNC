import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Function
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import Adam

# device = torch.cuda if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")

print(device)
# all data (push and pop) is input as a single tensor

class DNNC(Function):

    @staticmethod
    def forward(ctx, input, state):
        ctx.save_for_backward(input, state)
        output = state.clone()

        #binarise push and pop
        push = 0
        pop = 0
        threshold_push = 0.5
        threshold_pop = 0.5
        op = 'NoOp'
        if input[0]< threshold_push:
            push = 0
        elif input[0]>=threshold_push and input[0]<input[1]:
            push = 0
            pop = 1
            op = 'Pop'
        elif input[0]>=threshold_push and input[0]>=input[1]:
            push = 1
            pop = 0
            op='Push'
        if input[1]<threshold_pop:
            pop = 0
        elif input[1]>=threshold_pop and input[1]<input[0]:
            pop = 0
            push = 1
            op = 'Push'
        elif input[1]>=threshold_pop and input[1]>input[0]:
            pop = 1
            push = 0
            op = 'Pop'
        if push==0 and pop==0:
            op='NoOp'


        if op == 'Push':

            output[0]=state[0]+1

        elif op=='Pop':
            if state[0]>0:
                output[0] = state[0]-1

            elif state[0]==0:
                output[1]=state[1]+1

        elif op=='NoOp':

            pass

        ctx.op = op

        ignore = torch.tensor([0,0],dtype=torch.float32)

        return output





    @staticmethod
    def backward(ctx, grad_output):


        grad_output_stack_depth = grad_output[0].clone().detach()
        grad_output_falsepop = grad_output[1].clone().detach()
        grad_push_stack_depth = torch.tensor(1, dtype=torch.float32)
        grad_push_falsepop = torch.tensor(0, dtype=torch.float32)

        input, state = ctx.saved_tensors



        grad_input = grad_output.clone()


        if state[0]==0:
            grad_pop_stack_depth = torch.tensor(0, dtype=torch.float32)
            grad_pop_falsepop = torch.tensor(1, dtype=torch.float32)
        else:
            grad_pop_stack_depth = torch.tensor(-1, dtype=torch.float32)
            grad_pop_falsepop = torch.tensor(0, dtype=torch.float32)

        # multiply input gradients by output gradients and return the correct one (4 cases) and return 2 values

        grad_pop = (grad_pop_stack_depth*grad_input[0]) + (grad_pop_falsepop*grad_input[1])
        grad_push = (grad_push_stack_depth*grad_input[0]) + (grad_push_falsepop*grad_input[1])



        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        grad_stackdepth_out_stackdepth_in = 0
        grad_falsepop_out_falsepop_in = 0
        grad_stackdepth_out_falsepop_in = 0
        grad_falsepop_out_stackdepth_in = 0

        if ctx.op == 'Push':
            grad_stackdepth_out_stackdepth_in = 1
            grad_falsepop_out_falsepop_in = 1
            grad_stackdepth_out_falsepop_in = 0
            grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'NoOp':
            grad_stackdepth_out_stackdepth_in = 1
            grad_falsepop_out_falsepop_in = 1
            grad_stackdepth_out_falsepop_in = 0
            grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'Pop':
            if state[0]==0: #if stack is empty
                grad_stackdepth_out_stackdepth_in=0
                grad_falsepop_out_falsepop_in=1
                grad_falsepop_out_stackdepth_in=-1
                grad_stackdepth_out_falsepop_in=0
            elif state[0]>0: # if stack is not empty
                grad_stackdepth_out_stackdepth_in = 1
                grad_falsepop_out_falsepop_in = 1
                grad_stackdepth_out_falsepop_in = 0
                grad_falsepop_out_stackdepth_in = 0
        # grad_y = None
        grad_state = grad_output.clone()
        grad_state_stackdepth = (grad_stackdepth_out_stackdepth_in*grad_state[0])+(grad_falsepop_out_stackdepth_in*grad_state[1])
        grad_state_falsepop = (grad_stackdepth_out_falsepop_in*grad_state[0])+(grad_falsepop_out_falsepop_in*grad_state[1])
        grad_state = torch.tensor([grad_state_stackdepth,grad_state_falsepop],requires_grad=True)



        """
        test if the resulting gradients are correct
        this is only used for unit testing.
        comment lines between ######### if not unit testing 
        from this point up until the return (do not comment the return)
        """



        return grad_input, grad_state  # .to(device)

        """
        push = 1 --> grad_push = 1
        push = 0 --> grad_push = 0
        pop = 1, stack_depth>0, false_pop_count=0 --> grad_pop_stack_depth = -1 grad_pop_falsepop = 0
        pop = 1, stack_depth = 0, false_pop_count > 0 --> grad_stack_depth = 0, grad_pop_falsepop = 1


        """




class DNNCNN(nn.Module):
    def __init__(self):
    # def __init__(self, recurrent=False):
        super(DNNCNN, self).__init__()
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.dnnc = DNNC.apply


    def reset(self):
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)


    def forward(self, x,y=None):

        if y==None:
            y = torch.tensor([self.stack_depth, self.false_pop_count], requires_grad=True)

        # print('x input to DNNC ',x)
        # print('y input to DNNC ',y)
        x = self.dnnc(x,y)


        self.reset()
        self.stack_depth+=x[0]
        self.false_pop_count+=x[1]


        return x

    def editDNNCState(self, new_stack_depth, new_false_pop_count):
        if new_stack_depth<0:
            self.stack_depth = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        else:
            self.stack_depth = torch.tensor(new_stack_depth, dtype=torch.float32, requires_grad=False)
        if new_false_pop_count<0:
            self.false_pop_count = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        else:
            self.false_pop_count = torch.tensor(new_false_pop_count, dtype=torch.float32, requires_grad=False)




class DNNCNoFalsePop(Function):

    @staticmethod
    def forward(ctx, input, state):
        ctx.save_for_backward(input, state)
        output = state.clone()

        #binarise push and pop
        push = 0
        pop = 0
        threshold_push = 0.5
        threshold_pop = 0.5
        op = 'NoOp'
        if input[0]< threshold_push:
            push = 0
        elif input[0]>=threshold_push and input[0]<input[1]:
            push = 0
            pop = 1
            op = 'Pop'
        elif input[0]>=threshold_push and input[0]>=input[1]:
            push = 1
            pop = 0
            op='Push'
        if input[1]<threshold_pop:
            pop = 0
        elif input[1]>=threshold_pop and input[1]<input[0]:
            pop = 0
            push = 1
            op = 'Push'
        elif input[1]>=threshold_pop and input[1]>input[0]:
            pop = 1
            push = 0
            op = 'Pop'
        if push==0 and pop==0:
            op='NoOp'


        if op == 'Push':

            output[0]=state[0]+1

        elif op=='Pop':
            if state[0]>0:
                output[0] = state[0]-1

            elif state[0]==0:
                output[1]=state[1]+1

        elif op=='NoOp':

            pass

        ctx.op = op

        ignore = torch.tensor([0,0],dtype=torch.float32)

        return output





    @staticmethod
    def backward(ctx, grad_output):


        grad_output_stack_depth = grad_output.item()#.clone().detach()
        # grad_output_falsepop = grad_output[1].clone().detach()
        grad_push_stack_depth = torch.tensor(1, dtype=torch.float32)
        # grad_push_falsepop = torch.tensor(0, dtype=torch.float32)

        input, state = ctx.saved_tensors



        grad_input = grad_output.clone()


        if state.item()==0:
            grad_pop_stack_depth = torch.tensor(0, dtype=torch.float32)
            # grad_pop_falsepop = torch.tensor(1, dtype=torch.float32)
        else:
            grad_pop_stack_depth = torch.tensor(-1, dtype=torch.float32)
            # grad_pop_falsepop = torch.tensor(0, dtype=torch.float32)

        # multiply input gradients by output gradients and return the correct one (4 cases) and return 2 values

        grad_pop = (grad_pop_stack_depth*grad_input[0]) #+ (grad_pop_falsepop*grad_input[1])
        grad_push = (grad_push_stack_depth*grad_input[0]) #+ (grad_push_falsepop*grad_input[1])



        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        grad_stackdepth_out_stackdepth_in = 0
        # grad_falsepop_out_falsepop_in = 0
        # grad_stackdepth_out_falsepop_in = 0
        # grad_falsepop_out_stackdepth_in = 0

        if ctx.op == 'Push':
            grad_stackdepth_out_stackdepth_in = 1
            # grad_falsepop_out_falsepop_in = 1
            # grad_stackdepth_out_falsepop_in = 0
            # grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'NoOp':
            grad_stackdepth_out_stackdepth_in = 1
            # grad_falsepop_out_falsepop_in = 1
            # grad_stackdepth_out_falsepop_in = 0
            # grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'Pop':
            if state[0]==0: #if stack is empty
                grad_stackdepth_out_stackdepth_in=0
                # grad_falsepop_out_falsepop_in=1
                # grad_falsepop_out_stackdepth_in=-1
                # grad_stackdepth_out_falsepop_in=0
            elif state[0]>0: # if stack is not empty
                grad_stackdepth_out_stackdepth_in = 1
                # grad_falsepop_out_falsepop_in = 1
                # grad_stackdepth_out_falsepop_in = 0
                # grad_falsepop_out_stackdepth_in = 0
        # grad_y = None
        grad_state = grad_output.clone()
        grad_state_stackdepth = (grad_stackdepth_out_stackdepth_in*grad_state[0])#+(grad_falsepop_out_stackdepth_in*grad_state[1])
        # grad_state_falsepop = (grad_stackdepth_out_falsepop_in*grad_state[0])+(grad_falsepop_out_falsepop_in*grad_state[1])
        # grad_state = torch.tensor([grad_state_stackdepth,grad_state_falsepop],requires_grad=True)
        grad_state = torch.tensor([grad_state_stackdepth], requires_grad=True)

        """
        test if the resulting gradients are correct
        this is only used for unit testing.
        comment lines between ######### if not unit testing 
        from this point up until the return (do not comment the return)
        """



        return grad_input, grad_state  # .to(device)

        """
        push = 1 --> grad_push = 1
        push = 0 --> grad_push = 0
        pop = 1, stack_depth>0, false_pop_count=0 --> grad_pop_stack_depth = -1 grad_pop_falsepop = 0
        pop = 1, stack_depth = 0, false_pop_count > 0 --> grad_stack_depth = 0, grad_pop_falsepop = 1


        """




class DNNCNNNoFalsePop(nn.Module):
    def __init__(self):
    # def __init__(self, recurrent=False):
        super(DNNCNNNoFalsePop, self).__init__()
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.dnnc = DNNCNoFalsePop.apply


    def reset(self):
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)


    def forward(self, x,y=None):

        if y==None:
            # y = torch.tensor([self.stack_depth, self.false_pop_count], requires_grad=True)
            y = torch.tensor([self.stack_depth], requires_grad=True)

        # print('x input to DNNC ',x)
        # print('y input to DNNC ',y)
        x = self.dnnc(x,y)
        print('x after DNNC',x)


        self.reset()
        # self.stack_depth+=x[0]
        # self.false_pop_count+=x[1]
        self.stack_depth += x.item()


        return x

    def editDNNCState(self, new_stack_depth):
        if new_stack_depth<0:
            self.stack_depth = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        else:
            self.stack_depth = torch.tensor(new_stack_depth, dtype=torch.float32, requires_grad=False)


#Discrete Full Range Counter
class DFRC(Function):

    @staticmethod
    def forward(ctx, input, state):
        ctx.save_for_backward(input, state)
        output = state.clone()

        #binarise push and pop
        push = 0
        pop = 0
        threshold_push = 0.5
        threshold_pop = 0.5
        op = 'NoOp'
        if input[0]< threshold_push:
            push = 0
        elif input[0]>=threshold_push and input[0]<input[1]:
            push = 0
            pop = 1
            op = 'Pop'
        elif input[0]>=threshold_push and input[0]>=input[1]:
            push = 1
            pop = 0
            op='Push'
        if input[1]<threshold_pop:
            pop = 0
        elif input[1]>=threshold_pop and input[1]<input[0]:
            pop = 0
            push = 1
            op = 'Push'
        elif input[1]>=threshold_pop and input[1]>input[0]:
            pop = 1
            push = 0
            op = 'Pop'
        if push==0 and pop==0:
            op='NoOp'


        if op == 'Push':

            # output[0]=state[0]+1
            output = state + 1

        elif op=='Pop':
            # if state[0]>0:
            # output[0] = state[0]-1
            output = state - 1

            # elif state[0]==0:
            #     output[1]=state[1]+1

        elif op=='NoOp':

            pass

        ctx.op = op

        ignore = torch.tensor([0,0],dtype=torch.float32)

        return output





    @staticmethod
    def backward(ctx, grad_output):


        # grad_output_stack_depth = grad_output[0].clone().detach()
        # # grad_output_falsepop = grad_output[1].clone().detach()
        # grad_push_stack_depth = torch.tensor(1, dtype=torch.float32)
        # # grad_push_falsepop = torch.tensor(0, dtype=torch.float32)

        grad_output_stack_depth = grad_output.clone().detach()
        grad_push_stack_depth = torch.tensor(1, dtype=torch.float32)

        input, state = ctx.saved_tensors



        grad_input = grad_output.clone()


        if state==0:
            grad_pop_stack_depth = torch.tensor(0, dtype=torch.float32)
            # grad_pop_falsepop = torch.tensor(1, dtype=torch.float32)
        else:
            grad_pop_stack_depth = torch.tensor(-1, dtype=torch.float32)
            # grad_pop_falsepop = torch.tensor(0, dtype=torch.float32)

        # multiply input gradients by output gradients and return the correct one (4 cases) and return 2 values

        # grad_pop = (grad_pop_stack_depth*grad_input[0]) #+ (grad_pop_falsepop*grad_input[1])
        # grad_push = (grad_push_stack_depth*grad_input[0])# + (grad_push_falsepop*grad_input[1])
        grad_pop = (grad_pop_stack_depth * grad_input[0])  # + (grad_pop_falsepop*grad_input[1])
        grad_push = (grad_push_stack_depth * grad_input[0])  # + (grad_push_falsepop*grad_input[1])

        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        grad_stackdepth_out_stackdepth_in = 0
        # grad_falsepop_out_falsepop_in = 0
        # grad_stackdepth_out_falsepop_in = 0
        # grad_falsepop_out_stackdepth_in = 0

        if ctx.op == 'Push':
            grad_stackdepth_out_stackdepth_in = 1
            # grad_falsepop_out_falsepop_in = 1
            # grad_stackdepth_out_falsepop_in = 0
            # grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'NoOp':
            grad_stackdepth_out_stackdepth_in = 1
            # grad_falsepop_out_falsepop_in = 1
            # grad_stackdepth_out_falsepop_in = 0
            # grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'Pop':
            grad_stackdepth_out_stackdepth_in = 1
            # if state[0]==0: #if stack is empty
            #     grad_stackdepth_out_stackdepth_in=0
            #     grad_falsepop_out_falsepop_in=1
            #     grad_falsepop_out_stackdepth_in=-1
            #     grad_stackdepth_out_falsepop_in=0
            # elif state[0]>0: # if stack is not empty
            #     grad_stackdepth_out_stackdepth_in = 1
            #     grad_falsepop_out_falsepop_in = 1
            #     grad_stackdepth_out_falsepop_in = 0
            #     grad_falsepop_out_stackdepth_in = 0
        # grad_y = None
        grad_state = grad_output.clone()
        grad_state_stackdepth = (grad_stackdepth_out_stackdepth_in*grad_state[0])#+(grad_falsepop_out_stackdepth_in*grad_state[1])
        # grad_state_falsepop = (grad_stackdepth_out_falsepop_in*grad_state[0])+(grad_falsepop_out_falsepop_in*grad_state[1])
        # grad_state = torch.tensor([grad_state_stackdepth,grad_state_falsepop],requires_grad=True)
        grad_state = torch.tensor([grad_state_stackdepth], requires_grad=True)



        """
        test if the resulting gradients are correct
        this is only used for unit testing.
        comment lines between ######### if not unit testing 
        from this point up until the return (do not comment the return)
        """



        return grad_input, grad_state  # .to(device)

        """
        push = 1 --> grad_push = 1
        push = 0 --> grad_push = 0
        pop = 1, stack_depth>0, false_pop_count=0 --> grad_pop_stack_depth = -1 grad_pop_falsepop = 0
        pop = 1, stack_depth = 0, false_pop_count > 0 --> grad_stack_depth = 0, grad_pop_falsepop = 1


        """




class DFRCNN(nn.Module):
    def __init__(self):
    # def __init__(self, recurrent=False):
        super(DFRCNN, self).__init__()
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.dfrc = DFRC.apply


    def reset(self):
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)


    def forward(self, x,y=None):

        if y==None:
            # y = torch.tensor([self.stack_depth, self.false_pop_count], requires_grad=True)
            y = torch.tensor([self.stack_depth], requires_grad=True)

        # print('x input to DNNC ',x)
        # print('y input to DNNC ',y)
        x = self.dfrc(x,y)


        self.reset()
        self.stack_depth+=x[0]
        # self.false_pop_count+=x[1]


        return x

    def editDNNCState(self, new_stack_depth):
        self.stack_depth = torch.tensor(new_stack_depth, dtype=torch.float32, requires_grad=False)

        # if new_stack_depth<0:
        #     self.stack_depth = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        # else:
        #     self.stack_depth = torch.tensor(new_stack_depth, dtype=torch.float32, requires_grad=False)

