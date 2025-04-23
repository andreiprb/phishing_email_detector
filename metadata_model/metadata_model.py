from data_holder import EmailDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchinfo import summary
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
# Load the dataset

emails = [
    ".idea/dataSources/CEAS_08.csv",
    ".idea/dataSources/Nazario_5.csv",
    ".idea/dataSources/Nazario.csv",
    ".idea/dataSources/Nigerian_5.csv",
    ".idea/dataSources/Nigerian_Fraud.csv",
    ".idea/dataSources/SpamAssasin.csv"
]

label_positions = [
    "second_last",
    "second_last",
    "last",
    "second_last",
    "last",
    "second_last"
]

dataset = EmailDataset(emails, label_positions)

print(f"Number of emails: {len(dataset)}")

print()

batch_size = 32
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for data, label in train_loader:
    print(data.shape, label.shape)
    break

class SimpleNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.fully_connected = nn.Linear(input_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_connected(x)
        return x
    
model = SimpleNN(input_size=6).to(device)
print(model)
summary(model)

total_correct = 0
total_samples = 0
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    
    y_hat = model(x).squeeze()
    
    # Added this as a sort of softmax function
    clase = y_hat >= 0.5

    # print(f'Clase prezise: {clase}')
    # print(f'Clase corecte: {y}')
    # print(f'Clase corecte: {clase == y}')

    corect = clase == y

    num_corect = corect.sum().item()
    total_correct += num_corect
    total_samples += len(y)

    break


print(f'Total corect: {total_correct}')
print(f'Total samples: {total_samples}')
# Compute overall accuracy
acuratete = total_correct / total_samples
print(f'Total accuracy: {acuratete:.4f}')


#TESTING FUNCTION
@torch.no_grad()
def test(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Testeaza modelul pe datele furnizate de :param loader:
    :param model: model de regresie logistica binara
    :param loader: un dataloader care furnizeaza datele din setul peset care se testeaza
    :param device: pe ce device se afla modelul (cpu, gpu, tpu etc)
    :return: acuratetea de predictie
    """
    # initializare valori pt statistica
    correctly_classified = 0
    total_items = 0
    # cand se face testarea, modelul nu mai invata. El e trecut explicit in mod de testare
    model.eval()
    for x, y in loader:
        # trecem datele din setul de testare pe acelasi device ca si modelul
        x, y = x.to(device), y.to(device)
        
        # modelul prezice probabilitatile conditionate pentru minibatchul curent
        y_hat = model(x).squeeze()
        
        # predictia e clasa pozitiva daca probabilitatea e >=0.5, altfel clasa negativa
        predicted_class = y_hat >= 0.5
        
        correctly_classified += torch.sum(predicted_class == y)
        total_items += len(x)
    accuracy = correctly_classified / total_items
    return accuracy.cpu().detach().item()

acc = test(model, test_loader, device)
print(f'Acuratetea modelului neantrenat: {acc * 100}%')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss()

epochs = 50

losses = []
accuracies = []
train_accuracies = []

for epoch in range(epochs):
    # modelul trebuie trecut in modul train inainte de a se face instruirea lui
    # trecerea lui pe modul eval apare la apelul metodei de test()
    model.train()
    epoch_loss = 0
    total_items = 0
    for x, y in train_loader:
        # trecem datele din setul de antrenare pe acelasi device ca si modelul
        x, y = x.to(device), y.to(device)
        
        # stergem gradientii calculati anterior, altfel se face acumulare e gradienti - nu e de interes pt acest model
        model.zero_grad()
        
        # modelul prezice probabilitatile conditionate pentru minibatchul curent
        y_hat = torch.squeeze(model(x))
        
        # se calculeaza functia de eroare pe minibatchul curent
        loss = loss_fn(y_hat, y)
        # loss-ul calculat este media valorii de eroare peste minibatchul curent; 
        # inmultim media cu numarul de valori din minibatch pentru a determina valoarea cumulata 
        # a erorii pe minibatch
        epoch_loss += loss.item() * len(x)
        # actualizam numarul total de valori peset care s-a facut invatarea
        total_items += len(x)
        
        # cerem calcul de gradienti
        loss.backward()
        
        # optimizatorul aplica gradientii pe ponderi = invatare
        optimizer.step()
    
    epoch_loss /= total_items
    losses.append(epoch_loss)
    # afisam statistici
    print(f'Epoca: {epoch+1}/{epochs}: loss = {epoch_loss:.7f}')
    acc_test = test(model, test_loader, device)
    accuracies.append(acc_test)
    acc_train = test(model, train_loader, device)
    train_accuracies.append(acc_train)
    print(f'Epoca: {epoch + 1}/{epochs}: acuratete pe setul de antrenare = {acc_train * 100:.4f}%')
    print(f'Epoca: {epoch + 1}/{epochs}: acuratete pe setul de testare = {acc_test * 100:.4f}%\n')

plt.plot(losses)
plt.xlabel('Epoca')
plt.ylabel('Eroare')
plt.title(f'Valoarea functiei de eroare in timpul instruirii')
plt.show()

# a plot of the accuracies side by side
plt.plot(accuracies, label='Test')
plt.plot(train_accuracies, label='Train')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.title(f'Acuratetea in timpul instruirii')
plt.legend()
plt.show()