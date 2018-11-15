# Scale-Varying Triplet Ranking for Age Estimation
This is a Tensorflow implementation of ACCV2018 paper: [_Scale-Varying Triplet Ranking with
Classification Loss for Facial Age Estimation_](https://sgvr.kaist.ac.kr/publication/accv2018-age-estimation/).
The project is based on a pretrained model of [FaceNet](http://arxiv.org/abs/1503.03832), 
which is implemented and open-sourced by [davidsandberg's GitHub repo](https://github.com/davidsandberg/facenet).

## Training data
To train the network, you need to get your data ready for training.
The sample code includes and example when I used 
[MORPH dataset](https://ebill.uncw.edu/C20231_ustores/web/classic/store_main.jsp?STOREID=4)
that you can buy for $199.00 for academic use.

Otherwise, you may want to use other datasets including 
[Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html),
FG-NET, [ChaLearn](http://chalearnlap.cvc.uab.es/), etc.

`./data/morph.py` contains an example code of data preperation. You can easily
modify that code for other datasets.

## How to use it
1. Align face images using any of face detection work. I recommend you to use
[MTCNN](https://github.com/AITTSMD/MTCNN-Tensorflow) which performs really well
in most cases.
2. Prepare pretrained face recognition model. You need to convert pretrained model from
[this repo](https://github.com/davidsandberg/facenet) to numpy file to load it correctly.
I provide the [converted checkpoint file](http://sglab.kaist.ac.kr/~wbim/paper/accv2018-age-estimation/model-20170216-091149.ckpt-250000.npy).
Simply put the file under `./pretrained/FaceNet/20170216-091149/`
3. Connect your data by setting path conf file `./conf/path_default.conf`.
4. Set `log_dir` in `train_joint_morph_id.py`, and others if needed.
5. Run `train_joint_morph_id.py`

