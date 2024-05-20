import torch
import torch.optim as optim
from tqdm import tqdm
from eval import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCELoss()

def train(model, name, train_loader, num_epochs=30, topk=10, lr=0.001):
    print(f"Training model: {name}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    best_HR = 0; best_NDCG = 0
    best_model_state = None
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0
        for idx, (user_idxs, item_idxs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            user_idxs = user_idxs.to(device)
            item_idxs = item_idxs.to(device)
            labels = labels.float().to(device)

            outputs = model(user_idxs, item_idxs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        HR, NDCG = evaluate(model, topk)
        print(f'Epoch {epoch+1}/{num_epochs}, Batch loss: {running_loss/idx:.4f}, HR@{topk}: {HR:.4f}, NDCG@{topk}: {NDCG:.4f}')
        if HR > best_HR:
            best_HR = HR
            best_NDCG = NDCG
            best_model_state = model.state_dict()
    print(f"Training finish. Best HR@{topk}: {best_HR:.4f}, Best NDCG@{topk}: {best_NDCG:.4f}")
    torch.save(best_model_state.state_dict(), f'./model/best_{name}.pth')