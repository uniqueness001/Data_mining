# 另外一种方法是对训练基中的特征维度进行操作的，这次不是给每一个基分类器全部的特征，
# 而是给不同的基分类器分不同的特征，即比如基分类器1训练前半部分特征，
# 基分类器2训练后半部分特征（可以通过sklearn 的pipelines 实现）。最终通过StackingClassifier组合起来。
# encoding:utf-8
# @author:zee
from sklearn.datasets import load_iris
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X = iris.data
y = iris.target
pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)),
                      LogisticRegression())
pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),
                      LogisticRegression())
sclf = StackingClassifier(classifiers=[pipe1, pipe2],
                          meta_classifier=LogisticRegression())
sclf.fit(X, y)