from helper import downloadData
from src import dataAnalysis
from src import predictor

def main():
    downloadData.downloadCsv()
    dataAnalysis.plotting()
    predictor.predict()

if __name__ == '__main__':
    main()