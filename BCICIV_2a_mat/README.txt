{\rtf1\ansi\ansicpg950\cocoartf2638
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Preprocessing of BCICIV2a dataset:\
1. Retained the same frequency(0.5-100Hz)\
2. 4.5 seconds lond per trial\
3. Remove EOG channels\
4. 250 sampling rate\
\
Usage:\
For python lover:\
\
import scipy.io as io\
data = io.loadmat(DATA_PATH)\
\
\
www}