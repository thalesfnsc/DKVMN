import numpy as np
import math
import mxnet as mx
import mxnet.ndarray as nd
from sklearn import metrics

def norm_clipping(params_grad, threshold):
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        norm_val += np.sqrt(
            sum([nd.norm(grads[i]).asnumpy()[0] ** 2
                 for grads in params_grad]))
    norm_val /= float(len(params_grad[0]))

    if norm_val > threshold:
        ratio = threshold / float(norm_val)
        for grads in params_grad:
            for grad in grads:
                grad[:] *= ratio

    return norm_val


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log( np.maximum(1e-10,pred)) + \
           (1.0 - target) * np.log( np.maximum(1e-10, 1.0-pred) )
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)



def train(net, params, q_data, qa_data, label):
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T# Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        if params.show: bar.next()

        q_one_seq = q_data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_q = q_one_seq[:,:] # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx*params.batch_size:(idx+1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        #target = target.astype(np.int)
        #print(target)
        target = (target - 1) / params.n_question
        target = np.floor(target)
        #print(target)
        #target = target.astype(np.float) # correct: 1.0; wrong 0.0; padding -1.0

        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)

        data_batch = mx.io.DataBatch(data = [input_q, input_qa], label = [target])
        net.forward(data_batch, is_train=True)
        pred = net.get_outputs()[0].asnumpy() #(seqlen * batch_size, 1)
        net.backward()

        norm_clipping(net._exec_group.grad_arrays, params.maxgradnorm)
        net.update()

        target = target.asnumpy().reshape((-1,)) # correct: 1.0; wrong 0.0; padding -1.0

        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()

    all_pred = np.concatenate(pred_list,axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    print("all_target", all_target)
    print("all_pred", all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, q_data, qa_data, label, save_preds=False):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size))) #number of batches
    q_data = q_data.T  # Shape: (200,3633) #transpose it.
    qa_data = qa_data.T  # Shape: (200,3633) #transpose the answers
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    count = 0
    element_count = 0
    for idx in range(N):
        if params.show: bar.next()

        inds = np.arange(idx * params.batch_size, (idx + 1) * params.batch_size) #arange is an array version of Python's range. so we get indices for the items we're working on.
        q_one_seq = q_data.take(inds, axis=1, mode='wrap') #and we pull them out.
        qa_one_seq = qa_data.take(inds, axis=1, mode='wrap')
        #print 'seq_num', seq_num

        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)
        target = qa_one_seq[:, :]
        #target = target.astype(np.int)
        #target = (target - 1) / params.n_question
        #target = target.astype(np.float)  # correct: 1.0; wrong 0.0; padding -1.0
        
        target = (target - 1) / params.n_question
        target = np.floor(target)
        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)

        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[])
        net.forward(data_batch, is_train=False)
        pred = net.get_outputs()[0].asnumpy()
        teste = pred.reshape((params.seqlen,params.batch_size))[:,:params.batch_size]
        
        print(teste.shape)
        pred_transpose = teste.T
        
        print(pred_transpose[0])
        target = target.asnumpy()

        target_transpose = target.T
        print(target_transpose[0])
        #Concatenar o teste ao longo dos batches até somar os 598 alunos
        #Fazer o mesmo com o target transpose
        #Tentar gerar o knowledge estimates.


        break
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            target = target[:, :real_batch_size]
            pred = pred.reshape((params.seqlen, params.batch_size))[:, :real_batch_size]
            print(pred.shape)
            pred = pred.reshape((-1,))
            count += real_batch_size
        else:
            count += params.batch_size

        #here, target is 200*50, where 200 is problems and 50 is students (batched at 50, hence the 50)
        #hence, target[0] is a length-50 array of answers to problem number 1, for the relevant 50 students. answers as keyed below.
        target = target.reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0
        #this reshape just flattens it. It's now a length-10K array, in order. 50 answers for problem 1, then 50 for problem 2, etc.
        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist() #now we pull out the blanks.
        pred_nopadding = pred[nopadding_index] #the blanks should be the same in predictions and target so we use them for both.
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        #print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()
    assert count == seq_num

    print(len(pred_list[0]))
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    if save_preds:
        import pandas as pd
        pd.DataFrame(all_pred).to_csv("/content/DKVMN/data/all_pred.csv")
        pd.DataFrame(all_target).to_csv("/content/DKVMN/data/all_target.csv")
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc