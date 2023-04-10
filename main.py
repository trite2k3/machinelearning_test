### IMPORT ###
# Python ML
import torch
import torch.nn as nn

# Matte Library
import numpy as np

# Graf Library
import matplotlib.pyplot as plt

# HTTP Library
import requests

# Datetime Library
import time
### IMPORT ###



### FUNKTIONER ###
# Hämta datapunkter ifrån Yahoo Finance REST API
def get_stock_data(symbol):
    # Hämtar 10 datapunkter från en aktie på finance.yahoo.com

    # Skapa URL för att hämta data från Yahoo Finance
    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}'.format(symbol)

    # Ta datum
    now = int(time.time())
    # räkna 24 timmar sedan
    days_ago = 120  # Dagar att gå tillbaka

    seconds_in_day = 24 * 60 * 60  # Sekunder på en dag
    seconds_ago = days_ago * seconds_in_day  # Sekunder att gå tillbaka
    start = now - seconds_ago  # Unix timestamp

    # Parametrar för requests
    params = {
        'period1': start,
        'period2': now,
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': True,
    }
    # Behöver skicka mera HTTP headers!
    # Kör väl Mozilla eller hur var det?
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'DNT': '1',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Hämta HTML från webbplatsen
    response = requests.get(url, params=params, headers=headers)
    data = response.text

    return data
### FUNKTIONER ###



### SKAPA DATASET ###
# Testa datahämtning
symbol="GME"
datapoints=get_stock_data(symbol)

# Printa i format av request output
myarray = datapoints.replace("\n",",").split(",")
result_array = []
for i in range(0, len(myarray), 7):
    # Make array
    result_array.append(myarray[i:i + 7])
x_values = result_array
# Fältbeskrivningar
# print(x_values[0])

x_arr = []
y_arr = []
# Konvertera x-värdena till en tensor
# Loopa igenom datapunkter, fast skippa första då det bara är fältbeskrivningar
for i in range(1,len(x_values)):
    x_arr.append(x_values[i][2])
    y_arr.append(x_values[i][0])

# Skapa np.ndarray() av python list objektet (arrayen)
x_train = np.array(x_arr, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

# Konvertera y-värdena till en tensor
y_values = np.array(y_arr, dtype='datetime64[D]')

# Konvertera datumen till nummer sedan referensdatum (typ UNIX timestamp)
ref_date = np.datetime64('1970-01-01')
y_values = (y_values - ref_date).astype('timedelta64[D]').astype('int64')
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

# "Debug" lol
print(len(y_train))
print(y_train)
print(len(x_train))
print(x_train)
### SKAPA DATASET ###



### SKAPA MODELLEN ###
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Skapa en instans av modellen
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)
### SKAPA MODELLEN ###



### SKAPA FÖRLUST OCH OPTIMERING ###
# Skapa förlustfunktionen (Mean Squared Error)
criterion = nn.MSELoss()

# Skapa optimeringsfunktionen (Stochastic Gradient Descent)
# learning_rate = 0.01
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
### SKAPA FÖRLUST OCH OPTIMERING ###



### TRÄNA MODELLEN ###
# Träningsloop
epochs = 100

for epoch in range(epochs):
    # Konvertera data till tensors
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    # Räkna ut förutsägelserna (forward pass)
    outputs = model(inputs)

    # Beräkna förlusten (loss)
    loss = criterion(outputs, labels)

    # Nollställ optimeringsfunktionen (backward pass)
    optimizer.zero_grad()
    loss.backward()

    # Uppdatera parametrarna (weights)
    optimizer.step()

    # Skriv ut förlusten var 10:e epoch
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
### TRÄNA MODELLEN ###



### TESTA MODELLEN ###
# Testa modellen genom att förutsäga y-värden för nya x-värden
with torch.no_grad():
    predicted = model(torch.from_numpy(x_train)).detach().numpy()

# Oops...
print(predicted)

# Visualisera resultaten
# Har bytt plats på dem för vill ha datum på x-axeln
plt.plot(y_train, x_train, 'ro', label='Original data')
#plt.plot(predicted, x_train, label='Fitted line')
plt.plot(y_train, predicted, label='Fitted line')

# print(int(min(y_train)))
# print(int(max(y_train)))
# Set the y-axis limit to y_train
# Det här är bakvänt för jag ville ha datum på x axis xD
plt.xlim([int(min(y_train)), int(max(y_train))])
plt.ylim([int(min(x_train)), int(max(x_train))])

plt.legend()
plt.show()
### TESTA MODELLEN ###



