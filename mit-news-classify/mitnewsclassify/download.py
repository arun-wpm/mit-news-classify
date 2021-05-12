import os
import urllib
from urllib.request import urlretrieve
from tqdm import tqdm

# This is used to show progress when downloading.
# see here: https://github.com/tqdm/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download(model=None, google=False):
    # downloading from dropbox
    # maybe in the future allow for downloading only some models?
    urls = {
        # tfidf model
        "/data/tfidf/model_2500_500_50.h5":"https://www.dropbox.com/s/2t1bxjv7j4ssbaf/model_2500_500_50.h5?dl=1",
        "/data/tfidf/small_vocab_20.csv":"https://www.dropbox.com/s/v8ytgdeumwpw4g7/small_vocab_20.csv?dl=1",
        "/data/tfidf/tfmer_20.p":"https://www.dropbox.com/s/y46i525tzmf0hqx/tfmer_20.p?dl=1",
        '/data/tfidf/labelsdict_20.p':"https://www.dropbox.com/s/ioqoxawd5ekdl7c/labelsdict_20.p?dl=1",
        '/data/tfidf/nyt-theme-tags.csv':"https://www.dropbox.com/s/kvf34v0ecnc7lgc/nyt-theme-tags.csv?dl=1",
        # tfidf_bi model
        "/data/tfidf_bi/model_2000_500_50.h5":"https://www.dropbox.com/s/8j7dwpl8k1trcol/model_2000_500_50.h5?dl=1",
        "/data/tfidf_bi/small_vocab_bi_20.csv":"https://www.dropbox.com/s/ke2j923ryjriqeo/small_vocab_bi_20.csv?dl=1",
        "/data/tfidf_bi/tfmer_bi_20.p":"https://www.dropbox.com/s/tbdji109vtvdotm/tfmer_bi_20.p?dl=1",
        '/data/tfidf_bi/labelsdict_bi_20.p':"https://www.dropbox.com/s/20hx0oav37wf1ly/labelsdict_bi_20.p?dl=1",
        '/data/tfidf_bi/nyt-theme-tags.csv':"https://www.dropbox.com/s/4b1d8ej25af8j6q/nyt-theme-tags.csv?dl=1",
        # doc2vec model
        "/data/doc2vec/model_1200_800_40.h5":"https://www.dropbox.com/s/ooqavntcjia3ery/model_1200_800_40.h5?dl=1",
        "/data/doc2vec/doc2vec_model":"https://www.dropbox.com/s/hk4snqw43adxcog/doc2vec_model?dl=1",
        "/data/doc2vec/doc2vec_model.trainables.syn1neg.npy":"https://www.dropbox.com/s/7w9sv4bc7vuv30b/doc2vec_model.trainables.syn1neg.npy?dl=1",
        "/data/doc2vec/doc2vec_model.wv.vectors.npy":"https://www.dropbox.com/s/f2p3gp7v06qttc8/doc2vec_model.wv.vectors.npy?dl=1",
        "/data/doc2vec/labelsdict.p":"https://www.dropbox.com/s/33pcav9kjb8sh2f/labelsdict.p?dl=1",
        "/data/doc2vec/nyt-theme-tags.csv":"https://www.dropbox.com/s/sa2s8u4fyzeap5a/nyt-theme-tags.csv?dl=1",
        # gpt2 model
        "/data/gpt2/gpt_0.5.pth":"https://www.dropbox.com/s/r1icj5ytplfhuj0/gpt_0.5.pth?dl=1",
        "/data/gpt2/labels_dict_gpt.csv":"https://www.dropbox.com/s/vgtbf48q8deza9l/labels_dict_gpt.csv?dl=1",
        "/data/gpt2/nyt-theme-tags.csv":"https://www.dropbox.com/s/c1wts9knu3htzch/nyt-theme-tags.csv?dl=1",
        # distilbert model
        "/data/distilbert/labels_dict_distilbert.csv":"https://www.dropbox.com/s/tjnmd7z0uljaeg1/labels_dict_distilbert.csv?dl=1",
        "/data/distilbert/nyt-theme-tags.csv":"https://www.dropbox.com/s/omgstbndd3xl4cy/nyt-theme-tags.csv?dl=1",
        # ensemble model
        "/data/ensemble/model_ensemble.h5":"https://www.dropbox.com/s/pay4tvp6n2ffyhm/model_ensemble.h5?dl=1",
        "/data/ensemble/labelsdict.p":"https://www.dropbox.com/s/smrsyx96hf9nt9d/labelsdict.p?dl=1",
        "/data/ensemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/rprkzildvux9jjw/nyt-theme-tags.csv?dl=1",
        # trisemble model
        "/data/trisemble/model_trisemble.h5":"https://www.dropbox.com/s/gbgu53xetkfeiuk/model_trisemble.h5?dl=1",
        "/data/trisemble/labelsdict.p":"https://www.dropbox.com/s/9r2v169d2npbj9v/labelsdict.p?dl=1",
        "/data/trisemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/652kf8edq438bbo/nyt-theme-tags.csv?dl=1",
        # quadsemble model
        "/data/quadsemble/model_quadsemble.h5":"https://www.dropbox.com/s/p1jfocex3dqhrc9/model_quadsemble.h5?dl=1",
        "/data/quadsemble/labelsdict.p":"https://www.dropbox.com/s/sv14u7mwe2wnn49/labelsdict.p?dl=1",
        "/data/quadsemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/m94w28knnv2gcq9/nyt-theme-tags.csv?dl=1",
        # pentasemble model
        "/data/pentasemble/model_pentasemble.h5":"https://www.dropbox.com/s/jfyh18rmt36hqs5/model_pentasemble.h5?dl=1",
        "/data/pentasemble/labelsdict.p":"https://www.dropbox.com/s/orl0npue7zgtzzh/labelsdict.p?dl=1",
        "/data/pentasemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/1wdq5rs1dqu3bh6/nyt-theme-tags.csv?dl=1",
    }

    gurls = {
        # tfidf model
        "/gdata/tfidf/model_2500_500_50.h5":"https://www.dropbox.com/s/va0fns5mcavtpda/model_2500_500_50.h5?dl=1",
        "/gdata/tfidf/small_vocab_20.csv":"https://www.dropbox.com/s/oltdxb9jagt346y/small_vocab_20.csv?dl=1",
        "/gdata/tfidf/tfmer_20.p":"https://www.dropbox.com/s/36h7fhufsmfx0dj/tfmer_20.p?dl=1",
        '/gdata/tfidf/labelsdict_20.p':"https://www.dropbox.com/s/93j1i1pj0oehesp/labelsdict_20.p?dl=1",
        '/gdata/tfidf/nyt-theme-tags.csv':"https://www.dropbox.com/s/ndp06c3d834ywyp/nyt-theme-tags.csv?dl=1",
        # tfidf_bi model
        "/gdata/tfidf_bi/model_2000_500_50.h5":"https://www.dropbox.com/s/pkgu29v0krh6swf/model_2000_500_50.h5?dl=1",
        "/gdata/tfidf_bi/small_vocab_bi_20.csv":"https://www.dropbox.com/s/4aqrq69dbha4slc/small_vocab_bi_20.csv?dl=1",
        "/gdata/tfidf_bi/tfmer_bi_20.p":"https://www.dropbox.com/s/7xpri4qx5irow0p/tfmer_bi_20.p?dl=1",
        '/gdata/tfidf_bi/labelsdict_bi_20.p':"https://www.dropbox.com/s/9fepm04580mw65h/labelsdict_bi_20.p?dl=1",
        '/gdata/tfidf_bi/nyt-theme-tags.csv':"https://www.dropbox.com/s/n2jb4kdfbtq6eqv/nyt-theme-tags.csv?dl=1",
        # doc2vec model
        "/gdata/doc2vec/model_1200_800_40.h5":"https://www.dropbox.com/s/lzrti4x485ik751/model_1200_800_40.h5?dl=1",
        "/gdata/doc2vec/doc2vec_model":"https://www.dropbox.com/s/jip9eywghoq6osj/doc2vec_model?dl=1",
        "/gdata/doc2vec/doc2vec_model.trainables.syn1neg.npy":"https://www.dropbox.com/s/iva9oo993vrots8/doc2vec_model.trainables.syn1neg.npy?dl=1",
        "/gdata/doc2vec/doc2vec_model.wv.vectors.npy":"https://www.dropbox.com/s/cm5ytiej587s1m8/doc2vec_model.wv.vectors.npy?dl=1",
        "/gdata/doc2vec/labelsdict.p":"https://www.dropbox.com/s/b312d3cxgxc4fqs/labelsdict.p?dl=1",
        "/gdata/doc2vec/nyt-theme-tags.csv":"https://www.dropbox.com/s/a0zzwnkn6bugmhf/nyt-theme-tags.csv?dl=1",
        # gpt2 model
        # "/data/gpt2/gpt_0.5.pth":"https://www.dropbox.com/s/r1icj5ytplfhuj0/gpt_0.5.pth?dl=1",
        # "/data/gpt2/labels_dict_gpt.csv":"https://www.dropbox.com/s/vgtbf48q8deza9l/labels_dict_gpt.csv?dl=1",
        # "/data/gpt2/nyt-theme-tags.csv":"https://www.dropbox.com/s/c1wts9knu3htzch/nyt-theme-tags.csv?dl=1",
        # distilbert model
        "/gdata/distilbert/labels_dict_distilbert.csv":"https://www.dropbox.com/s/c1kps71rb12qeyi/labels_dict_distilbert.csv?dl=1",
        "/gdata/distilbert/nyt-theme-tags.csv":"https://www.dropbox.com/s/sbv57m5d69uzagl/nyt-theme-tags.csv?dl=1",
        # ensemble model
        "/gdata/ensemble/model_ensemble.h5":"https://www.dropbox.com/s/4yefrdwp2wk4ln4/model_ensemble.h5?dl=1",
        "/gdata/ensemble/labelsdict.p":"https://www.dropbox.com/s/0t25fdbff78u7eh/labelsdict.p?dl=1",
        "/gdata/ensemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/swuwh9yq2m0tsiy/nyt-theme-tags.csv?dl=1",
        # trisemble model
        "/gdata/trisemble/model_trisemble.h5":"https://www.dropbox.com/s/kmkri9kupviiohx/model_trisemble.h5?dl=1",
        "/gdata/trisemble/labelsdict.p":"https://www.dropbox.com/s/q6zh3eb663bjez3/labelsdict.p?dl=1",
        "/gdata/trisemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/q6keehvp6o5qlyo/nyt-theme-tags.csv?dl=1",
        # quadsemble model
        "/gdata/quadsemble/model_quadsemble.h5":"https://www.dropbox.com/s/d2iu7xhevvon6wz/model_quadsemble.h5?dl=1",
        "/gdata/quadsemble/labelsdict.p":"https://www.dropbox.com/s/0vcav8g2cwmih2k/labelsdict.p?dl=1",
        "/gdata/quadsemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/i1p5miapu4ugw1g/nyt-theme-tags.csv?dl=1",
        # pentasemble model
        # "/data/pentasemble/model_pentasemble.h5":"https://www.dropbox.com/s/jfyh18rmt36hqs5/model_pentasemble.h5?dl=1",
        # "/data/pentasemble/labelsdict.p":"https://www.dropbox.com/s/orl0npue7zgtzzh/labelsdict.p?dl=1",
        # "/data/pentasemble/nyt-theme-tags.csv":"https://www.dropbox.com/s/1wdq5rs1dqu3bh6/nyt-theme-tags.csv?dl=1",
    }

    # get package directory
    pwd = os.path.dirname(os.path.abspath(__file__))
    print("Package directory: " + pwd)

    # make directories as needed
    datadir = "/data" if google == False else "/gdata"
    try:
        os.mkdir(pwd + datadir)
    except FileExistsError:
        print(pwd + datadir + " directory already exists, some other models downloaded. Continuing...")
    
    dirs = [
        "/data/tfidf",
        "/data/tfidf_bi",
        "/data/doc2vec",
        "/data/gpt2",
        "/data/distilbert",
        "/data/ensemble",
        "/data/trisemble",
        "/data/quadsemble",
        "/data/pentasemble",
    ]
    gdirs = [
        "/gdata/tfidf",
        "/gdata/tfidf_bi",
        "/gdata/doc2vec",
        "/gdata/gpt2",
        "/gdata/distilbert",
        "/gdata/ensemble",
        "/gdata/trisemble",
        "/gdata/quadsemble",
        "/gdata/pentasemble",
    ]
    focusdir = dirs if google == False else gdirs
    for dir in focusdir:
        if (model is None or model == dir.split("/")[-1]):
            try:
                os.mkdir(pwd + dir)
            except FileExistsError:
                print(pwd + dir + " directory already exists... perhaps you already downloaded the data? Overwriting...")

    # download the files
    focusurls = urls if google == False else gurls
    for sink, source in focusurls.items():
        if (model is None or model == sink.split("/")[-2]):
            print("Downloading " + sink + " from " + source)
            try:
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=source.split('/')[-1]) as t:  # all optional kwargs
                    urlretrieve(source, filename=pwd+sink, reporthook=t.update_to, data=None)
                    t.total = t.n
            except urllib.error.ContentTooShortError:
                print("Download incomplete? Please try again.")