# encoding = utf-8
#@author zee(GDUT)
#读取原始数据集
import pandas as pd
data = pd.read_csv('D:/trains.csv',nrows=5)
data.head()
print(data)
print (data.info())
#删除严重缺失的数据
clean_data = data.drop( ['annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m','open_il_6m','open_il_12m'
                        ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc',
                        'all_util','inq_fi','total_cu_tl','inq_last_12m','desc','issue_d','pymnt_plan','application_type'],axis = 1)
#print (clean_data.info(n=30))
#进行众数填充，并输出新的数据集
clean_data1 = clean_data.fillna(clean_data.mode().iloc[0],inplace=True)
#clean_data1.head()
print(clean_data1.head(2))
clean_data1.to_csv('D:/newtrain1.csv',encoding='utf-8')
#检测单个数据的异常点
#dd = clean_data1.query()
#isAnomaly
#重新读取填充后的数据集
import pandas as pd
data = pd.read_csv('D:/newtrain1.csv')
#data.head()
data['int_rate'].value_counts()
print(data.head(50))
#print(data.info())
## 连续特征规范化处理
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
sca00=StandardScaler()
sca01=MinMaxScaler(feature_range=(0, 1))
sca02=Normalizer(norm='l2')
data['dti']=MinMaxScaler(feature_range=(0, 1)).fit_transform(data['dti'].values.reshape(-1,1))
data['dti']=data['dti'].rank()
print(data['dti'])
data['installment']=MinMaxScaler().fit_transform(data['dti'].values.reshape(-1,1))
data['installment']=data['installment'].rank()
print(data['installment'])
data['mths_since_last_record']=StandardScaler().fit_transform(data['mths_since_last_record'].values.reshape(-1,1))
data['mths_since_last_record']=data['mths_since_last_record'].rank()
data['revol_util']=MinMaxScaler().fit_transform(data['revol_util'].values.reshape(-1,1))
data['revol_util']=data['revol_util'].rank()
data['total_acc']=StandardScaler().fit_transform(data['total_acc'].values.reshape(-1,1))
data['total_acc']=data['total_acc'].rank()
#print(data.dtypes)
#类别特征处理
data01=pd.get_dummies(data,columns=['grade'],dummy_na=True)  #获得哑变量
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
#enc = OneHotEncoder
data['grade']=LabelEncoder().fit_transform(data['grade'].values)
data['sub_grade']=LabelEncoder().fit_transform(data['sub_grade'].values)
data['emp_title']=LabelEncoder().fit_transform(data['emp_title'].values)
data['home_ownership']=LabelEncoder().fit_transform(data['home_ownership'].values)
data['term']=LabelEncoder().fit_transform(data['term'].values)
data['emp_length']=LabelEncoder().fit_transform(data['emp_length'].values)
data['verification_status']=LabelEncoder().fit_transform(data['verification_status'].values)
data['loan_status']=LabelEncoder().fit_transform(data['loan_status'].values)
data['purpose']=LabelEncoder().fit_transform(data['purpose'].values)
data['title']=LabelEncoder().fit_transform(data['title'].values)
data['member_id']=LabelEncoder().fit_transform(data['member_id'].values)
data['initial_list_status']=LabelEncoder().fit_transform(data['initial_list_status'].values)
#data=LabelEncoder().fit_transform(data['home_ownership'].values)
#data01_a = enc.fit(data['grade'])
data01_a= OneHotEncoder().fit_transform(data['grade'].values.reshape(-1,1))
data01_b= OneHotEncoder().fit_transform(data['sub_grade'].values.reshape(-1,1))
data01_c= OneHotEncoder().fit_transform(data['emp_title'].values.reshape(-1,1))
data01_d= OneHotEncoder().fit_transform(data['home_ownership'].values.reshape(-1,1))
data01_e= OneHotEncoder().fit_transform(data['term'].values.reshape(-1,1))
data01_f= OneHotEncoder().fit_transform(data['emp_length'].values.reshape(-1,1))
data01_g= OneHotEncoder().fit_transform(data['verification_status'].values.reshape(-1,1))
data01_h= OneHotEncoder().fit_transform(data['loan_status'].values.reshape(-1,1))
data01_i= OneHotEncoder().fit_transform(data['purpose'].values.reshape(-1,1))
data01_j= OneHotEncoder().fit_transform(data['title'].values.reshape(-1,1))
data01_k= OneHotEncoder().fit_transform(data['member_id'].values.reshape(-1,1))
data01_l= OneHotEncoder().fit_transform(data['initial_list_status'].values.reshape(-1,1))
#data01_aa = float(data01_a)
data01=sparse.hstack([data,data01_a],dtype='object')
print(data01)
data02=sparse.hstack([data,data01_b],dtype='object')
#print(data02)
data03=sparse.hstack([data,data01_c],dtype='object')
#print(data03)
data04=sparse.hstack([data,data01_d],dtype='object')
#print(data04)
data05=sparse.hstack([data,data01_e],dtype='object')
data06=sparse.hstack([data,data01_f],dtype='object')
data07=sparse.hstack([data,data01_g],dtype='object')
data08=sparse.hstack([data,data01_h],dtype='object')
data09=sparse.hstack([data,data01_i],dtype='object')
data10=sparse.hstack([data,data01_j],dtype='object')
data11=sparse.hstack([data,data01_k],dtype='object')
data12=sparse.hstack([data,data01_l],dtype='object')
#data['grade'].value_counts()
#data['funded_amnt_inv'].value_counts().rank()
#文本向量化
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
#data['emp_length']=str(data['emp_length'])
data['funded_amnt_inv']=str(data['funded_amnt_inv']) #三介转化成两介
#data['emp_length']=str(data['emp_length'])
#data['emp_length']=data['emp_length'].apply(lambda x:' '.join(x.split(';')))
data['funded_amnt_inv']=data['funded_amnt_inv'].apply(lambda x:' '.join(x.split(';')))
#emp_length = CountVectorizer().fit_transform(data['emp_length'])
funded_amnt_inv = CountVectorizer().fit_transform(data['funded_amnt_inv'])
#data_a=sparse.hstack((emp_length,data),dtype='object')
data_b=sparse.hstack((funded_amnt_inv,data),dtype='object')
#print(data_b)
#print(data_a)
#data['annual_inc'].value_counts().rank()
data['loan_amnt'].value_counts().rank()
#data['funded_amnt'].value_counts().rank()
#data['funded_amnt_inv'].value_counts().rank()
#data['revol_bal'].value_counts().rank()
#data['tot_cur_bal'].value_counts().rank()
#data['total_rev_hi_lim'].value_counts().rank()
#data['out_prncp'].value_counts().rank()
#data['out_prncp_inv'].value_counts().rank()
#data['total_pymnt'].value_counts().rank()
#data['total_pymnt_inv'].value_counts().rank()
#data['total_rec_prncp'].value_counts().rank()
#data['total_rec_int'].value_counts().rank()
###连续特征离散化
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data['annual_inc']=pd.cut(data['annual_inc'],bins=[40000,80000,120000,160000,200000]).astype('str')
data['annual_inc']=LabelEncoder().fit_transform(data['annual_inc'])
data['annual_inc'].head(n=30)