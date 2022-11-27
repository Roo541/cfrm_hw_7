import numpy as np
import pandas as pd

df = pd.read_csv('fama_french.csv')

r_f = df['rf'].values
r_gm = df['gm'].values - r_f
r_adbe = df['adbe'].values - r_f
r_ora = df['ora'].values - r_f
r_flo = df['flo'].values - r_f
f_m = df['mkt_rf'].values
f_smb = df['smb'].values
f_hml = df['hml'].values
vector_1 = [1]*len(f_m)

R = np.matrix([r_gm, r_adbe, r_ora, r_flo])
R = np.transpose(R)
F = np.matrix([vector_1, f_m, f_smb, f_hml])
F = np.transpose(F)

x1 = np.dot(np.transpose(F), F)
x2 = np.linalg.inv(x1)
x3 = np.dot(x2, np.transpose(F))
B = np.dot(x3,R)

sample = {'sample':['alpha', 'beta_1', 'beta_2', 'beta_3'], 'r_gm':[], 'r_adbe':[], 'r_ora':[], 'r_flo':[]}
v = np.array(B)
#part a
for i in range(len(v)):
    sample['r_gm'].append(v[i][0])
    sample['r_adbe'].append(v[i][1])
    sample['r_ora'].append(v[i][2])
    sample['r_flo'].append(v[i][3])
sample = pd.DataFrame.from_dict(sample)
sample = sample.set_index('sample')
print('Part (a.)')
print('B hat Matrix: ')
print(sample)

#part b
E_hat = np.array(R) - np.dot(np.array(F),np.array(B))
corr_matrix_e = np.zeros((4,4))
corr_matrix_r = np.zeros((4,4))

alpha = np.transpose(E_hat)
excess = {'e_gm':alpha[0], 'e_adbe':alpha[1], 'e_ora':alpha[2], 'e_flo':alpha[3]}
dh = pd.DataFrame.from_dict(excess)
#residuals
i = 0
for l in dh.keys():
    corr_matrix_e[i][0] = dh[l].corr(dh['e_gm'])
    corr_matrix_e[i][1] = dh[l].corr(dh['e_adbe'])
    corr_matrix_e[i][2] = dh[l].corr(dh['e_ora'])
    corr_matrix_e[i][3] = dh[l].corr(dh['e_flo'])
    i += 1
#excess returns
i = 0
dr = df
dr['gm'] = dr['gm'] - dr['rf']
dr['adbe'] = dr['adbe'] - dr['rf']
dr['ora'] = dr['ora'] - dr['rf']
dr['flo'] = dr['flo'] - dr['rf']
for l in dr.keys():
    if l in ['gm', 'adbe', 'ora', 'flo']:
        df[l] = df[l] - df['rf']
        corr_matrix_r[i][0] = dr[l].corr(dr['gm'])
        corr_matrix_r[i][1] = dr[l].corr(df['adbe'])
        corr_matrix_r[i][2] = dr[l].corr(df['ora'])
        corr_matrix_r[i][3] = dr[l].corr(df['flo'])
        i += 1

sample = {'':['gm', 'adbe', 'ora', 'flo'], 'gm':[], 'adbe':[], 'ora':[], 'flo':[]}
v = corr_matrix_e
for i in range(len(v)):
    sample['gm'].append(v[i][0])
    sample['adbe'].append(v[i][1])
    sample['ora'].append(v[i][2])
    sample['flo'].append(v[i][3])
sample = pd.DataFrame.from_dict(sample)
sample_e = sample.set_index('')

sample = {'':['gm', 'adbe', 'ora', 'flo'], 'gm':[], 'adbe':[], 'ora':[], 'flo':[]}
v = corr_matrix_r
for i in range(len(v)):
    sample['gm'].append(v[i][0])
    sample['adbe'].append(v[i][1])
    sample['ora'].append(v[i][2])
    sample['flo'].append(v[i][3])
sample = pd.DataFrame.from_dict(sample)
sample_r = sample.set_index('')

print('')
print('part (b.)')
print('sample correlation matrix of the residuals:')
print(sample_e)
print('')
print('sample correlation matrix of excess returns:')
print(sample_r)

#part c
#make covariance matrix
cov_matrix_r = np.zeros((4,4))
i = 0
for l in df.keys():
    if l in ['gm', 'adbe', 'ora', 'flo']:
        cov_matrix_r[i][0] = df[l].cov(df['gm'])
        cov_matrix_r[i][1] = df[l].cov(df['adbe'])
        cov_matrix_r[i][2] = df[l].cov(df['ora'])
        cov_matrix_r[i][3] = df[l].cov(df['flo'])
        i += 1

sample = {'':['gm', 'adbe', 'ora', 'flo'], 'gm':[], 'adbe':[], 'ora':[], 'flo':[]}
v = cov_matrix_r
for i in range(len(v)):
    sample['gm'].append(v[i][0])
    sample['adbe'].append(v[i][1])
    sample['ora'].append(v[i][2])
    sample['flo'].append(v[i][3])
sample = pd.DataFrame.from_dict(sample)
sample_Sigma = sample.set_index('')

print('')
print('part (c.)')
print('sample covariance matrix of R:')
print(sample_Sigma)
print('')
print(cov_matrix_r)
print('')
e_vals, e_vectors = np.linalg.eig(cov_matrix_r)
print('Eigen Values:')
print('')
print(e_vals)
print('')
print('Eigen Vectors:')
print(e_vectors)

