import time
import datetime
import torch
import math
from utils import log_string,metric
from utils import load_data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

def train(model, args, log, loss_criterion, optimizer, scheduler):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY,  mean, std) = load_data(args)
    num_train, _, _ = trainX.shape
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)
    model = model.to(device)

    val_loss_min = float('inf')
    train_total_loss = []
    val_total_loss = []

    # Train & validation
    for epoch in range(args.max_epoch):
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        # train
        start_train = time.time()
        model.train()
        train_loss = 0
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            label = trainY[start_idx: end_idx]
            X,TE = X.to(device),TE.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(X, TE)
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch) * (end_idx - start_idx)
            loss_batch.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (batch_idx+1) % 400 == 0:
                print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del X, TE, label, pred, loss_batch
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # val loss
        start_val = time.time()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                label = valY[start_idx: end_idx]
                X,TE = X.to(device),TE.to(device)
                label = label.to(device)
                pred = model(X, TE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
                del X, TE, label, pred, loss_batch
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            val_loss_min = val_loss
            torch.save(model, args.model_file)

        #test loss
        model.eval()
        testPred = []
        with torch.no_grad():
            for batch_idx in range(test_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
                X = testX[start_idx: end_idx]
                TE = testTE[start_idx: end_idx]
                X,TE = X.to(device),TE.to(device)
                t1 = time.time()
                pred_batch = model(X, TE)
                testPred.append(pred_batch.cpu().detach().clone())
                del X, TE, pred_batch
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
        test_mae, test_rmse, test_mape = metric(testPred, testY)
        log_string(log, 'test             mae %.2f\t\trmse %.2f\t\tmape %.2f%%' %
                    (test_mae, test_rmse, test_mape * 100))
        scheduler.step()

    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss
