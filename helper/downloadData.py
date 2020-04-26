import requests

def downloadCsv():
    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
    dataFile = requests.get(url)
    dataContent = dataFile.content
    csv_file = open('data/dataCsv.csv', 'wb')
    csv_file.write(dataContent)
    csv_file.close()
