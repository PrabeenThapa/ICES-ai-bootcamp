# Pandas Cheatsheet
0. **Install**: `pip install pandas`

## Basics
1. **Import Pandas**: `import pandas as pd`
2. **Create DataFrame**: `df = pd.DataFrame(data)`
3. **Create Series**: `s = pd.Series(data)`
4. **Read CSV**: `pd.read_csv('filename.csv')`
5. **Read Excel**: `pd.read_excel('filename.xlsx')`
6. **Read JSON**: `pd.read_json('filename.json')`
7. **Read SQL**: `pd.read_sql('query', conn)`
8. **Read HTML**: `pd.read_html('url')`
9. **Read HDF5**: `pd.read_hdf('filename.h5', key='key')`
10. **Read Parquet**: `pd.read_parquet('filename.parquet')`
11. **Read Feather**: `pd.read_feather('filename.feather')`
12. **Read Msgpack**: `pd.read_msgpack('filename.msg')` *(deprecated, use `read_pickle`)*
13. **Read Stata**: `pd.read_stata('filename.dta')`
14. **Read SAS**: `pd.read_sas('filename.sas7bdat')`
15. **Read Clipboards**: `pd.read_clipboard()`
16. **Write CSV**: `df.to_csv('filename.csv')`
17. **Write Excel**: `df.to_excel('filename.xlsx')`
18. **Write JSON**: `df.to_json('filename.json')`
19. **Write SQL**: `df.to_sql('table_name', conn)`
20. **Write HDF5**: `df.to_hdf('filename.h5', key='key')`
21. **Write Parquet**: `df.to_parquet('filename.parquet')`
22. **Write Feather**: `df.to_feather('filename.feather')`
23. **Write Msgpack**: `df.to_msgpack('filename.msg')` *(deprecated, use `to_pickle`)*
24. **Write Stata**: `df.to_stata('filename.dta')`
25. **Write SAS**: `df.to_sas('filename.sas7bdat')`

## DataFrame Basics
26. **Display DataFrame**: `df.head()` (first 5 rows)
27. **Tail**: `df.tail()` (last 5 rows)
28. **Info**: `df.info()`
29. **Describe**: `df.describe()`
30. **Shape**: `df.shape`
31. **Columns**: `df.columns`
32. **Index**: `df.index`
33. **Data types**: `df.dtypes`
34. **Memory usage**: `df.memory_usage()`
35. **Rename columns**: `df.rename(columns={'old': 'new'})`
36. **Set index**: `df.set_index('column')`
37. **Reset index**: `df.reset_index()`
38. **Drop column**: `df.drop('column', axis=1)`
39. **Drop row**: `df.drop(index)`
40. **Reorder columns**: `df = df[['col1', 'col2', 'col3']]`

## Series Basics
41. **Create Series**: `pd.Series(data)`
42. **Indexing**: `s[0]` or `s['index_label']`
43. **Slice**: `s[1:4]`
44. **Operations**: `s + 5`
45. **Apply function**: `s.apply(func)`
46. **Map function**: `s.map(func)`
47. **Value counts**: `s.value_counts()`
48. **Unique values**: `s.unique()`
49. **Check for nulls**: `s.isnull()`
50. **Fill nulls**: `s.fillna(value)`

## Indexing & Selection
51. **Select column**: `df['column']`
52. **Select rows by index**: `df.loc[0]`
53. **Select rows by condition**: `df[df['column'] > value]`
54. **Select specific rows and columns**: `df.loc[0:2, ['col1', 'col2']]`
55. **Select using iloc**: `df.iloc[0:2, 0:2]`
56. **Select with multiple conditions**: `df[(df['col1'] > value) & (df['col2'] < value)]`
57. **Query method**: `df.query('col1 > value')`
58. **Use isin**: `df[df['column'].isin([value1, value2])]`
59. **Use between**: `df[df['column'].between(start, end)]`
60. **Select with index names**: `df.loc[['index1', 'index2']]`
61. **Select with index positions**: `df.iloc[[0, 2]]`

## Data Cleaning
62. **Drop duplicates**: `df.drop_duplicates()`
63. **Remove NaN values**: `df.dropna()`
64. **Fill NaN values**: `df.fillna(value)`
65. **Replace values**: `df.replace(to_replace, value)`
66. **Convert to numeric**: `pd.to_numeric(df['column'], errors='coerce')`
67. **Convert to datetime**: `pd.to_datetime(df['column'])`
68. **Check for missing values**: `df.isnull().sum()`
69. **Forward fill**: `df.ffill()`
70. **Backward fill**: `df.bfill()`
71. **Apply string method**: `df['column'].str.method()`

## Aggregation & GroupBy
72. **Group by column**: `df.groupby('column')`
73. **Aggregation functions**: `df.groupby('column').sum()`
74. **Multiple aggregation functions**: `df.groupby('column').agg(['sum', 'mean'])`
75. **Pivot table**: `df.pivot_table(values='value', index='index', columns='columns')`
76. **Cross-tabulation**: `pd.crosstab(df['col1'], df['col2'])`
77. **Transform**: `df.groupby('column').transform(lambda x: x - x.mean())`
78. **Filter groups**: `df.groupby('column').filter(lambda x: len(x) > 10)`
79. **Resample**: `df.resample('M').mean()`
80. **Rolling window**: `df['column'].rolling(window=3).mean()`
81. **Expanding window**: `df['column'].expanding().mean()`

## Merging & Joining
82. **Merge DataFrames**: `pd.merge(df1, df2, on='key')`
83. **Left join**: `pd.merge(df1, df2, how='left', on='key')`
84. **Right join**: `pd.merge(df1, df2, how='right', on='key')`
85. **Outer join**: `pd.merge(df1, df2, how='outer', on='key')`
86. **Inner join**: `pd.merge(df1, df2, how='inner', on='key')`
87. **Concatenate DataFrames**: `pd.concat([df1, df2], axis=0)`
88. **Append DataFrames**: `df1.append(df2)`
89. **Join DataFrames on index**: `df1.join(df2, how='left')`
90. **Combine data**: `df1.combine_first(df2)`
91. **Combine with filling**: `df1.combine(df2, lambda x, y: x.fillna(y))`

## Date & Time
92. **Current date**: `pd.Timestamp.now()`
93. **Create date range**: `pd.date_range(start='2024-01-01', end='2024-12-31')`
94. **Create period range**: `pd.period_range(start='2024-01', end='2024-12', freq='M')`
95. **Convert to datetime**: `pd.to_datetime(df['column'])`
96. **Extract year**: `df['date'].dt.year`
97. **Extract month**: `df['date'].dt.month`
98. **Extract day**: `df['date'].dt.day`
99. **Extract weekday**: `df['date'].dt.weekday()`
100. **Extract day of week name**: `df['date'].dt.day_name()`
101. **Extract hour**: `df['date'].dt.hour`
102. **Extract minute**: `df['date'].dt.minute`
103. **Date arithmetic**: `df['date'] + pd.DateOffset(days=1)`
104. **Date difference**: `df['end_date'] - df['start_date']`
105. **Resample by month**: `df.resample('M').mean()`
106. **Shift dates**: `df['date'].shift(1)`

## Data Transformation
107. **Apply function**: `df.apply(func)`
108. **Apply lambda function**: `df.apply(lambda x: x + 1)`
109. **Map function**: `df['column'].map(func)`
110. **Apply with axis**: `df.apply(func, axis=1)`
111. **Transform function**: `df.transform(func)`
112. **Aggregate function**: `df.agg(func)`
113. **Sort values**: `df.sort_values(by='column')`
114. **Sort index**: `df.sort_index()`
115. **Rank values**: `df['column'].rank()`
116. **Replace values**: `df.replace(to_replace, value)`
117. **Rename index**: `df.rename_axis('new_name')`
118. **Apply across axis**: `df.apply(func, axis=0)`

## Visualization
119. **Basic plot**: `df.plot()`
120. **Histogram**: `df['column'].hist()`
121. **Box plot**: `df.boxplot(column='column')`
122. **Scatter plot**: `df.plot.scatter(x='col1', y='col2')`
123. **Line plot**: `df.plot.line(x='x', y='y')`
124. **Bar plot**: `df.plot.bar(x='x', y='y')`
125. **Pie chart**: `df['column'].plot.pie()`
126. **Hexbin plot**: `df.plot.hexbin(x='x', y='y', gridsize=30)`
127. **Density plot**: `df.plot.kde()`
128. **Subplots**: `df.plot(subplots=True)`

## DataFrame Manipulation
129. **Set column names**: `df.columns = ['col1', 'col2']`
130. **Reset index**: `df.reset_index(drop=True)`
131. **Rename columns**: `df.rename(columns={'old': 'new'})`
132. **Fill missing values**: `df.fillna(value)`
133. **Drop rows with missing values**: `df.dropna()`
134. **Drop columns with missing values**: `df.dropna(axis=1)`
135. **Get unique values**: `df['column'].unique()`
136. **Get value counts**: `df['column'].value_counts()`
137. **Count occurrences**: `df['column'].count()`
138. **Convert to list**: `df['column'].tolist()`
139. **Drop specific values**: `df[df['column'] != value]`
140. **Find duplicates**: `df.duplicated()`
141. **Drop duplicates**: `df.drop_duplicates()`
142. **Change data type**: `df['column'] = df['column'].astype(int)`
143. **Copy DataFrame**: `df.copy()`
144. **Transpose DataFrame**: `df.T`
145. **Stack DataFrame**: `df.stack()`
146. **Unstack DataFrame**: `df.unstack()`
147. **Pivot**: `df.pivot(index='index', columns='columns', values='values')`
148. **Melt**: `pd.melt(df, id_vars=['id'], value_vars=['value1', 'value2'])`
149. **Set column as index**: `df.set_index('column')`
150. **Get item by index**: `df.iloc[index]`
151. **Get item by label**: `df.loc[label]`
152. **Check for duplicates**: `df.duplicated()`
153. **Sort values by multiple columns**: `df.sort_values(by=['col1', 'col2'])`

## Advanced Indexing
154. **MultiIndex**: `pd.MultiIndex.from_product([list1, list2])`
155. **Set MultiIndex**: `df.set_index(['col1', 'col2'])`
156. **Reset MultiIndex**: `df.reset_index()`
157. **Slice MultiIndex**: `df.loc[('label1', 'label2')]`
158. **Stack MultiIndex**: `df.stack()`
159. **Unstack MultiIndex**: `df.unstack()`
160. **Get level values**: `df.index.get_level_values(level)`
161. **Swap levels**: `df.swaplevel()`
162. **Reorder levels**: `df.reorder_levels([1, 0])`

## Window Functions
163. **Rolling mean**: `df['column'].rolling(window=3).mean()`
164. **Rolling sum**: `df['column'].rolling(window=3).sum()`
165. **Expanding mean**: `df['column'].expanding().mean()`
166. **Expanding sum**: `df['column'].expanding().sum()`
167. **EWM mean**: `df['column'].ewm(span=3).mean()`
168. **EWM variance**: `df['column'].ewm(span=3).var()`

## Pivoting
169. **Pivot Table**: `df.pivot_table(values='value', index='index', columns='columns', aggfunc='mean')`
170. **Cross-tab**: `pd.crosstab(df['col1'], df['col2'])`
171. **Stack**: `df.stack()`
172. **Unstack**: `df.unstack()`

## Combining DataFrames
173. **Concatenate**: `pd.concat([df1, df2])`
174. **Join**: `df1.join(df2, how='left')`
175. **Merge**: `pd.merge(df1, df2, on='key')`
176. **Append**: `df1.append(df2)`

## Handling Missing Data
177. **Drop rows with missing data**: `df.dropna()`
178. **Fill missing data with value**: `df.fillna(value)`
179. **Forward fill missing data**: `df.ffill()`
180. **Backward fill missing data**: `df.bfill()`
181. **Replace missing values**: `df.replace(np.nan, value)`
182. **Interpolate missing values**: `df.interpolate()`

## File Handling
183. **Read from clipboard**: `pd.read_clipboard()`
184. **Write to clipboard**: `df.to_clipboard()`
185. **Write DataFrame to Excel**: `df.to_excel('filename.xlsx')`
186. **Write DataFrame to CSV**: `df.to_csv('filename.csv')`
187. **Write DataFrame to HDF5**: `df.to_hdf('filename.h5', key='key')`
188. **Read from Excel**: `pd.read_excel('filename.xlsx')`
189. **Read from CSV**: `pd.read_csv('filename.csv')`
190. **Read from HDF5**: `pd.read_hdf('filename.h5', key='key')`

## Working with Text Data
191. **Split strings**: `df['column'].str.split('delimiter')`
192. **Replace strings**: `df['column'].str.replace('old', 'new')`
193. **Extract substring**: `df['column'].str.slice(start, stop)`
194. **Check for substring**: `df['column'].str.contains('substring')`
195. **Convert to lowercase**: `df['column'].str.lower()`
196. **Convert to uppercase**: `df['column'].str.upper()`
197. **Remove whitespace**: `df['column'].str.strip()`
198. **Pad strings**: `df['column'].str.pad(width, side='left')`
199. **Format strings**: `df['column'].str.format()`
200. **Count occurrences**: `df['column'].str.count('substring')`

## DataFrame Construction
201. **Create DataFrame from dict**: `pd.DataFrame(dict)`
202. **Create DataFrame from list of dicts**: `pd.DataFrame([dict1, dict2])`
203. **Create DataFrame from list of tuples**: `pd.DataFrame([(1, 2), (3, 4)], columns=['A', 'B'])`
204. **Create Series from dict**: `pd.Series(dict)`
205. **Create Series from list**: `pd.Series([1, 2, 3])`

## Performance
206. **Check DataFrame size**: `df.memory_usage()`
207. **Optimize data types**: `df['column'] = df['column'].astype('category')`
208. **Use categories**: `df['column'] = pd.Categorical(df['column'])`
209. **Use efficient data formats**: `df.to_parquet('filename.parquet')`

## Date & Time
210. **Get current date and time**: `pd.Timestamp.now()`
211. **Create date range**: `pd.date_range(start='2024-01-01', periods=10)`
212. **Create time series**: `pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')`
213. **Convert to period**: `df['date'].dt.to_period('M')`
214. **Date arithmetic**: `df['date'] + pd.DateOffset(days=10)`
215. **Extract time**: `df['date'].dt.time`
216. **Extract date**: `df['date'].dt.date`

## Resampling
217. **Resample time series**: `df.resample('M').mean()`
218. **Resample with different frequency**: `df.resample('A').sum()`
219. **Resample with multiple functions**: `df.resample('W').agg({'col1': 'mean', 'col2': 'sum'})`
220. **Downsample**: `df.resample('D').mean()`
221. **Upsample**: `df.resample('H').ffill()`

## Handling Duplicates
222. **Find duplicates**: `df.duplicated()`
223. **Drop duplicates**: `df.drop_duplicates()`
224. **Keep first duplicate**: `df.drop_duplicates(keep='first')`
225. **Keep last duplicate**: `df.drop_duplicates(keep='last')`

## Statistical Functions
226. **Mean**: `df['column'].mean()`
227. **Median**: `df['column'].median()`
228. **Standard deviation**: `df['column'].std()`
229. **Variance**: `df['column'].var()`
230. **Min**: `df['column'].min()`
231. **Max**: `df['column'].max()`
232. **Sum**: `df['column'].sum()`
233. **Count**: `df['column'].count()`
234. **Correlation**: `df.corr()`
235. **Covariance**: `df.cov()`

## Advanced Usage
236. **Apply custom function**: `df.apply(lambda x: custom_function(x))`
237. **Use query for complex conditions**: `df.query('column1 > value1 & column2 < value2')`
238. **Set MultiIndex**: `df.set_index(['col1', 'col2'])`
239. **Flatten MultiIndex DataFrame**: `df.reset_index()`
240. **Pivot with multiple values**: `df.pivot_table(values=['val1', 'val2'], index='index', columns='columns')`
241. **Explode lists in cells**: `df.explode('column')`
242. **Combine data with fill**: `df1.combine_first(df2)`
243. **Query with variables**: `df.query('column1 > @value1')`
244. **Get item by label**: `df.loc['label']`
245. **Get item by position**: `df.iloc[0]`
246. **Assign new values**: `df.assign(new_col=lambda x: x['col1'] + x['col2'])`
247. **Use `applymap` for elementwise operations**: `df.applymap(lambda x: x*2)`
248. **Use `transform` for broadcasting**: `df.transform(lambda x: x - x.mean())`
249. **Using `pivot` for wide format**: `df.pivot(index='index', columns='columns', values='values')`
250. **Using `melt` for long format**: `pd.melt(df, id_vars=['id'], value_vars=['value1', 'value2'])`
