from bs4 import BeautifulSoup
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import requests
from tqdm.notebook import tqdm

# download and clean data

# ChIP-seq

## download
def _mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def chip_seq(task):
    '''download ChIP-seq data'''
    if task not in {'motif_discovery', 'motif_occupancy'}:
        raise ValueError(f'task can only be in {tasks}, got \'{task}\'')
    r = requests.get(f'http://cnn.csail.mit.edu/{task}/')
    soup = BeautifulSoup(r.text)
    trs = soup.find('table').find_all('tr')[3:-1]
    folders = [tr.a.text for tr in trs]
    mkdir(task)

    for folder in tqdm(folders):
        _mkdir(os.path.join(task, folder))
        for data in ['train.data', 'test.data']:
            r = requests.get(f'http://cnn.csail.mit.edu/{task}/{folder}/{data}')
            with open(os.path.join(task, folder, data), 'w') as f:
                f.write(r.text)

## transform
def load_chip_seq_as_df(path, file):
    '''load the downloaded text files into a single DataFrame'''
    dfs = []
    for folder in tqdm(os.listdir(path)):
        try:
            df = pd.read_csv(os.path.join(path, folder, file), sep=' ', header=None)
            dfs.append(df)
        except:
            print(f'Skip {folder}')
            continue
    result = pd.concat(dfs)
    result.sort_index(inplace=True)
    return result

def df_to_fasta(df, file, data_dir):
    '''dump the DataFrame as a fasta file, skip sequenecs that have N'''
    gen = (
        SeqRecord(Seq(record[1]), id='', name='', description=str(record[2]))
        for idx, record in df.iterrows()
        if not 'N' in record[1]
    )
    with open(data_dir + file, 'w') as f:
        SeqIO.write(tqdm(gen), f, 'fasta')

# histone

## download
def histone():
    '''download histone data'''
    links = [1, 2, 4, 5, 8, 10, 11, 12, 13, 14]
    files = ['H3',      'H4',      'H3K9ac',  'H3K14ac',  'H4ac',
            'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K36me3', 'H3K79me3']
    files = [file + '.fasta' for file in files]

    for link, file in zip(links, files):
        r = requests.get(f'http://www.jaist.ac.jp/~tran/nucleosome/ten_dataset/dataset{link:02}.txt')
        with open(file, 'w') as f:
            f.write(r.text)

## transform
def clean(record):
    '''separate the sequence and label'''
    seq = record.seq._data
    return SeqRecord(Seq(seq[:-1]), id='', name='', description=str(seq[-1]))

def clean_histone(files, files_):
    '''clean the histone fasta file'''
    for i in tqdm(range(len(files))):
        with open(files[i], 'r') as f:
            records = [clean(record) for record in SeqIO.parse(f, 'fasta')
                        if len(record) == 501]
        with open(files_[i], 'w') as f:
            SeqIO.write(records, f, 'fasta')