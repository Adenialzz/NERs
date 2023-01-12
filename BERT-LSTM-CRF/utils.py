
def demask(mask, labels, preds):
    labels_ll, preds_ll = [], []
    for i, sent_mask in enumerate(mask):
        labels_sent, preds_sent = [], []
        for j, word_mask in enumerate(sent_mask):
            if word_mask == 0:
                continue
            labels_sent.append(labels[i][j].item())
            preds_sent.append(preds[i][j].item())
        labels_ll.append(labels_sent)
        preds_ll.append(preds_sent)
    return labels_ll, preds_ll
