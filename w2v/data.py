import json
import random
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def n_vocab():
    with open('data/vocab.json') as f:
        token_to_ix = json.loads(f.read())
        return len(token_to_ix)


class W2VDataset(Dataset):

    def __init__(self):
        with open('data/vocab.json') as f:
            token_to_ix = json.loads(f.read())
        self.vocab = list(token_to_ix.keys())

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int) -> str:
        return self.vocab[item]


class W2VDataLoader(DataLoader):

    def __init__(self,
                 batch_size: int,
                 n_negs: int):
        dataset = W2VDataset()
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.collate)
        self.n_negs = n_negs
        with open('data/contexts.json') as f:
            self.token_to_context = json.loads(f.read())
        with open('data/vocab.json') as f:
            self.token_to_ix = json.loads(f.read())
        self.vocab = list(self.token_to_ix.keys())

    def collate(self, batch: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare a batch.

        Returns:
            targets: torch.LongTensor, of shape (n_batch, 1), vocab indices for
              target words.
            contexts: torch.LongTensor, of shape (n_batch, 1), vocab indices for
              context words.
            negatives: torch.LongTensor, of shape(n_batch, n_negs), vocab
              indices for negative samples.
        """
        # gather contexts and negative samples
        targets = batch
        contexts = []
        negatives = []

        for target in targets:
            # context word
            ctxs = self.token_to_context[target]
            try:
                ctx = random.choice(ctxs)
            except Exception as e:
                print(ctxs)
                raise e
            contexts.append(ctx)

            # negative samples
            negs = []
            while len(negs) < self.n_negs:
                neg = random.choice(self.vocab)
                if neg not in self.token_to_context[target] + [target]:
                    negs.append(neg)
            negatives.append(negs)

        # convert to ids
        targets = [self.token_to_ix[t] for t in targets]
        contexts = [self.token_to_ix[t] for t in contexts]
        negatives = [[self.token_to_ix[t] for t in n] for n in negatives]

        # convert to long tensors
        targets = Tensor(targets).long().unsqueeze(-1)
        contexts = Tensor(contexts).long().unsqueeze(-1)
        negatives = Tensor(negatives).long()

        return targets, contexts, negatives
