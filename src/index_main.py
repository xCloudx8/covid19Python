from helper import downloadData
from src import dataAnalysis

def main():
    downloadData.downloadCsv()
    dataAnalysis.plotting()

if __name__ == '__main__':
    main()