# Numpy Cheatsheet

## Prerequisite of Numpy  
0. **Install Numpy**: `pip install numpy`
1. **Import Numpy**: `import numpy as np`

## Array Creation
2. **From list**: `np.array([1, 2, 3])`
3. **From tuple**: `np.array((1, 2, 3))`
4. **Zeros array**: `np.zeros((3, 4))`
5. **Ones array**: `np.ones((2, 3))`
6. **Constant array**: `np.full((2, 2), 7)`
7. **Identity matrix**: `np.eye(3)`
8. **Diagonal matrix**: `np.diag([1, 2, 3])`
9. **Arange**: `np.arange(10, 30, 5)`
10. **Linspace**: `np.linspace(0, 2, 9)`
11. **Random array**: `np.random.random((2, 2))`
12. **Random integers**: `np.random.randint(0, 10, (3, 3))`
13. **Random normal**: `np.random.normal(0, 1, (3, 3))`
14. **Random uniform**: `np.random.uniform(0, 1, (3, 3))`
15. **Empty array**: `np.empty((2, 2))`
16. **From buffer**: `np.frombuffer(b'hello world', dtype='S1')`
17. **From function**: `np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)`
18. **From iterable**: `np.fromiter(range(5), dtype=int)`
19. **From string**: `np.fromstring('1 2 3', sep=' ')`
20. **Meshgrid**: `np.meshgrid(np.arange(3), np.arange(3))`
21. **Zeros_like**: `np.zeros_like(arr)`
22. **Ones_like**: `np.ones_like(arr)`
23. **Empty_like**: `np.empty_like(arr)`
24. **Full_like**: `np.full_like(arr, 9)`
25. **Complex numbers**: `np.array([1+2j, 3+4j])`

## Array Attributes
26. **Shape**: `arr.shape`
27. **Number of dimensions**: `arr.ndim`
28. **Size (number of elements)**: `arr.size`
29. **Data type**: `arr.dtype`
30. **Item size**: `arr.itemsize`
31. **Total bytes consumed**: `arr.nbytes`
32. **Base object**: `arr.base`
33. **Memory layout**: `arr.flags`
34. **Byteswap**: `arr.byteswap()`
35. **Strides**: `arr.strides`
36. **Itemsize of elements**: `arr.itemsize`

## Array Operations
37. **Element-wise addition**: `arr1 + arr2`
38. **Element-wise subtraction**: `arr1 - arr2`
39. **Element-wise multiplication**: `arr1 * arr2`
40. **Element-wise division**: `arr1 / arr2`
41. **Element-wise power**: `arr1 ** arr2`
42. **Dot product**: `np.dot(arr1, arr2)`
43. **Matrix multiplication**: `arr1 @ arr2`
44. **Transpose**: `arr.T`
45. **Inverse**: `np.linalg.inv(arr)`
46. **Determinant**: `np.linalg.det(arr)`
47. **Eigenvalues & Eigenvectors**: `np.linalg.eig(arr)`
48. **Solve linear equations**: `np.linalg.solve(arr1, arr2)`
49. **Kronecker product**: `np.kron(arr1, arr2)`
50. **Clip values**: `np.clip(arr, a_min, a_max)`
51. **Round**: `np.round(arr)`
52. **Floor**: `np.floor(arr)`
53. **Ceil**: `np.ceil(arr)`
54. **Truncate**: `np.trunc(arr)`
55. **Add at**: `np.add.at(arr, indices, values)`
56. **Subtract at**: `np.subtract.at(arr, indices, values)`
57. **Multiply at**: `np.multiply.at(arr, indices, values)`
58. **Divide at**: `np.divide.at(arr, indices, values)`
59. **Power at**: `np.power.at(arr, indices, values)`
60. **Set at**: `np.put(arr, indices, values)`
61. **Get from positions**: `np.take(arr, indices)`

## Universal Functions (ufuncs)
62. **Square root**: `np.sqrt(arr)`
63. **Exponentiation**: `np.exp(arr)`
64. **Logarithm**: `np.log(arr)`
65. **Log base 10**: `np.log10(arr)`
66. **Log base 2**: `np.log2(arr)`
67. **Sine**: `np.sin(arr)`
68. **Cosine**: `np.cos(arr)`
69. **Tangent**: `np.tan(arr)`
70. **Arc sine**: `np.arcsin(arr)`
71. **Arc cosine**: `np.arccos(arr)`
72. **Arc tangent**: `np.arctan(arr)`
73. **Hyperbolic sine**: `np.sinh(arr)`
74. **Hyperbolic cosine**: `np.cosh(arr)`
75. **Hyperbolic tangent**: `np.tanh(arr)`
76. **Absolute value**: `np.abs(arr)`
77. **Sign**: `np.sign(arr)`
78. **Log1p (log(1 + x))**: `np.log1p(arr)`
79. **Expm1 (exp(x) - 1)**: `np.expm1(arr)`
80. **Degrees**: `np.degrees(arr)`
81. **Radians**: `np.radians(arr)`
82. **Isnan**: `np.isnan(arr)`
83. **Isfinite**: `np.isfinite(arr)`
84. **Isinf**: `np.isinf(arr)`
85. **Iscomplex**: `np.iscomplex(arr)`
86. **Isreal**: `np.isreal(arr)`
87. **Iscomplexobj**: `np.iscomplexobj(arr)`
88. **Isrealobj**: `np.isrealobj(arr)`

## Aggregation Functions
89. **Sum**: `np.sum(arr)`
90. **Minimum**: `np.min(arr)`
91. **Maximum**: `np.max(arr)`
92. **Mean**: `np.mean(arr)`
93. **Median**: `np.median(arr)`
94. **Standard Deviation**: `np.std(arr)`
95. **Variance**: `np.var(arr)`
96. **Product**: `np.prod(arr)`
97. **Cumulative sum**: `np.cumsum(arr)`
98. **Cumulative product**: `np.cumprod(arr)`
99. **Range (ptp)**: `np.ptp(arr)`
100. **Non-zero elements**: `np.count_nonzero(arr)`
101. **All elements true**: `np.all(arr)`
102. **Any element true**: `np.any(arr)`
103. **Array equality**: `np.array_equal(arr1, arr2)`
104. **Array equivalence**: `np.array_equiv(arr1, arr2)`

## Indexing & Slicing
105. **Single element**: `arr[0, 1]`
106. **Slice**: `arr[0:2]`
107. **Advanced slicing**: `arr[0:2, 1:3]`
108. **Boolean indexing**: `arr[arr > 5]`
109. **Fancy indexing**: `arr[[1, 3, 5]]`
110. **Non-contiguous slicing**: `arr[::2]`
111. **Negative indexing**: `arr[-1]`
112. **Field access**: `arr['field']` (for structured arrays)
113. **Field assignment**: `arr['field'] = value`
114. **Array split**: `np.array_split(arr, sections)`

## Reshaping
115. **Reshape**: `arr.reshape((3, 4))`
116. **Flatten**: `arr.flatten()`
117. **Ravel**: `arr.ravel()`
118. **Transpose**: `arr.T`
119. **Resize**: `np.resize(arr, (2, 3))`
120. **Swap axes**: `np.swapaxes(arr, axis1, axis2)`
121. **Move axis**: `np.moveaxis(arr, source, destination)`
122. **Roll axis**: `np.rollaxis(arr, axis, start)`
123. **Resize array in-place**: `np.resize(arr, new_shape)`
124. **Squeeze dimensions**: `np.squeeze(arr)`
125. **Expand dimensions**: `np.expand_dims(arr, axis)`

## Stacking
126. **Vertical Stack**: `np.vstack((arr1, arr2))`
127. **Horizontal Stack**: `np.hstack((arr1, arr2))`
128. **Column Stack**: `np.column_stack((arr1, arr2))`
129. **Depth Stack**: `np.dstack((arr1, arr2))`
130. **Concatenate**: `np.concatenate((arr1, arr2), axis=0)`
131. **Stack along new axis**: `np.stack((arr1, arr2), axis=0)`
132. **Block matrix**: `np.block([[arr1, arr2], [arr3, arr4]])`

## Splitting
133. **Array split**: `np.split(arr, sections)`
134. **Horizontal split**: `np.hsplit(arr, sections)`
135. **Vertical split**: `np.vsplit(arr, sections)`
136. **Depth split**: `np.dsplit(arr, sections)`
137. **Array split by indices**: `np.array_split(arr, indices)`

## Linear Algebra
138. **Dot product**: `np.dot(arr1, arr2)`
139. **Matrix product**: `np.matmul(arr1, arr2)`
140. **Matrix power**: `np.linalg.matrix_power(arr, n)`
141. **Solve triangular**: `np.linalg.solve_triangular(arr1, arr2)`
142. **Inner product**: `np.inner(arr1, arr2)`
143. **Outer product**: `np.outer(arr1, arr2)`
144. **Tensor dot**: `np.tensordot(arr1, arr2, axes)`
145. **Matrix rank**: `np.linalg.matrix_rank(arr)`

## Random
146. **Random sample**: `np.random.sample((3, 3))`
147. **Random choice**: `np.random.choice(arr, size, replace=True)`
148. **Random permutation**: `np.random.permutation(arr)`
149. **Random shuffle**: `np.random.shuffle(arr)`
150. **Random seed**: `np.random.seed(seed)`
151. **Random bytes**: `np.random.bytes(10)`

## Broadcasting
152. **Broadcast arrays**: `np.broadcast(arr1, arr2)`
153. **Tile array**: `np.tile(arr, reps)`
154. **Repeat elements**: `np.repeat(arr, repeats)`
155. **Tile array with shape**: `np.broadcast_to(arr, shape)`

## Save & Load
156. **Save array**: `np.save('filename.npy', arr)`
157. **Load array**: `np.load('filename.npy')`
158. **Save multiple arrays**: `np.savez('filename.npz', arr1=arr1, arr2=arr2)`
159. **Load multiple arrays**: `np.load('filename.npz')`
160. **Save text**: `np.savetxt('filename.txt', arr)`
161. **Load text**: `np.loadtxt('filename.txt')`
162. **Savez_compressed**: `np.savez_compressed('filename.npz', arr1=arr1, arr2=arr2)`

## Miscellaneous
163. **Apply along axis**: `np.apply_along_axis(func, axis, arr)`
164. **Apply over multiple axes**: `np.apply_over_axes(func, arr, axes)`
165. **Vectorize a function**: `np.vectorize(func)(arr)`
166. **Piecewise function**: `np.piecewise(arr, condlist, funclist)`
167. **Histogram**: `np.histogram(arr, bins)`
168. **Digitize**: `np.digitize(arr, bins)`
169. **Histogram2d**: `np.histogram2d(x, y, bins)`
170. **Histogramdd**: `np.histogramdd([x, y, z], bins)`
171. **Bin count**: `np.bincount(arr)`
172. **Covariance matrix**: `np.cov(arr)`
173. **Correlation coefficient**: `np.corrcoef(arr)`
174. **Percentile**: `np.percentile(arr, q)`
175. **Quantile**: `np.quantile(arr, q)`
176. **Average with weights**: `np.average(arr, weights=weights)`
177. **Meshgrid with sparse**: `np.meshgrid(arr1, arr2, sparse=True)`
178. **Mgrid**: `np.mgrid[0:5, 0:5]`
179. **Ogrid**: `np.ogrid[0:5, 0:5]`
180. **Ravel multi-index**: `np.ravel_multi_index((1, 2), (3, 3))`
181. **Unravel index**: `np.unravel_index(5, (3, 3))`
182. **View array with new dtype**: `arr.view(np.uint8)`
183. **Shared memory flag**: `np.may_share_memory(arr1, arr2)`
184. **Find indices of max values**: `np.argmax(arr, axis)`
185. **Find indices of min values**: `np.argmin(arr, axis)`
186. **Searchsorted**: `np.searchsorted(arr, values)`
187. **Resize array in-place**: `np.resize(arr, new_shape)`
188. **Array properties**: `np.info(arr)`
189. **Check type**: `np.issubdtype(arr.dtype, np.float)`
190. **Memory usage**: `arr.nbytes`
191. **Repeat elements**: `np.repeat(arr, repeats)`
192. **Unique elements**: `np.unique(arr)`
193. **Sort array**: `np.sort(arr)`
194. **Lexsort**: `np.lexsort((arr1, arr2))`
195. **Meshgrid**: `np.meshgrid(arr1, arr2)`
196. **Tile array**: `np.tile(arr, reps)`
197. **Invert elements**: `np.invert(arr)`
198. **Left shift**: `np.left_shift(arr1, arr2)`
199. **Right shift**: `np.right_shift(arr1, arr2)`
200. **Roll elements**: `np.roll(arr, shift)`
201. **Set element**: `arr.flat[index] = value`
202. **Get element**: `arr.flat[index]`
203. **Item size in bytes**: `arr.itemsize`
204. **Base array**: `arr.base`
205. **Flags**: `arr.flags`
206. **Fill array**: `arr.fill(value)`
207. **Set element to scalar**: `np.put(arr, indices, value)`
208. **Linear index to multi index**: `np.unravel_index(indices, arr.shape)`
209. **Multi index to linear index**: `np.ravel_multi_index(multi_indices, arr.shape)`
210. **Byteswap array**: `arr.byteswap()`
211. **Get field**: `arr.getfield(dtype, offset)`
212. **Set field**: `arr.setfield(val, dtype, offset)`
213. **Repeat array**: `np.repeat(arr, repeats, axis)`
214. **Indices of non-zero elements**: `np.nonzero(arr)`
215. **Select elements based on condition**: `np.where(condition, x, y)`
216. **Count non-zero elements**: `np.count_nonzero(arr)`
217. **Flatten multi-dimensional array**: `arr.flat`
218. **Convert to matrix**: `np.asmatrix(arr)`
219. **Return a view**: `arr.view()`
220. **Update array inplace**: `arr[...] = value`
221. **In-place operation**: `np.add(arr1, arr2, out=arr)`
222. **Identity matrix**: `np.identity(n)`
223. **Eye matrix**: `np.eye(N, M, k=0)`
224. **Triangular matrix**: `np.tri(N, M, k=0)`
225. **Tril**: `np.tril(arr)`
226. **Triu**: `np.triu(arr)`
227. **Set operations**: `np.union1d(arr1, arr2)`
228. **Set intersection**: `np.intersect1d(arr1, arr2)`
229. **Set difference**: `np.setdiff1d(arr1, arr2)`
230. **Set exclusive or**: `np.setxor1d(arr1, arr2)`
231. **Find common elements**: `np.in1d(arr1, arr2)`
232. **Find unique elements**: `np.unique(arr)`
233. **Check sorted**: `np.all(np.diff(arr) >= 0)`
234. **Find index of max element**: `np.argmax(arr)`
235. **Find index of min element**: `np.argmin(arr)`
236. **Search for element**: `np.searchsorted(arr, value)`
237. **Find peaks**: `np.argmax(arr, axis=0)`
238. **Calculate log sum**: `np.logaddexp(arr1, arr2)`
239. **Calculate log sum exp2**: `np.logaddexp2(arr1, arr2)`
240. **Generate polynomial**: `np.polynomial.polynomial.Polynomial([coeffs])`
241. **Fit polynomial**: `np.polynomial.polynomial.polyfit(x, y, deg)`
242. **Evaluate polynomial**: `np.polynomial.polynomial.polyval(x, p)`
243. **Get polynomial roots**: `np.polynomial.polynomial.polyroots(p)`
244. **Polynomial derivative**: `np.polynomial.polynomial.polyder(p)`
245. **Polynomial integral**: `np.polynomial.polynomial.polyint(p)`
246. **Broadcasted comparison**: `np.less(arr1, arr2)`
247. **Broadcasted equality**: `np.equal(arr1, arr2)`
248. **Broadcasted greater**: `np.greater(arr1, arr2)`
249. **Broadcasted less equal**: `np.less_equal(arr1, arr2)`
250. **Broadcasted greater equal**: `np.greater_equal(arr1, arr2)`

