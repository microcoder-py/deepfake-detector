
Ö§
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8Ö±

deepfake_detector/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name deepfake_detector/dense/kernel

2deepfake_detector/dense/kernel/Read/ReadVariableOpReadVariableOpdeepfake_detector/dense/kernel*
_output_shapes

:*
dtype0

deepfake_detector/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namedeepfake_detector/dense/bias

0deepfake_detector/dense/bias/Read/ReadVariableOpReadVariableOpdeepfake_detector/dense/bias*
_output_shapes
:*
dtype0
¶
)deepfake_detector/time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)deepfake_detector/time_distributed/kernel
¯
=deepfake_detector/time_distributed/kernel/Read/ReadVariableOpReadVariableOp)deepfake_detector/time_distributed/kernel*&
_output_shapes
: *
dtype0
¦
'deepfake_detector/time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'deepfake_detector/time_distributed/bias

;deepfake_detector/time_distributed/bias/Read/ReadVariableOpReadVariableOp'deepfake_detector/time_distributed/bias*
_output_shapes
: *
dtype0
¬
'deepfake_detector/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 Ø@*8
shared_name)'deepfake_detector/lstm/lstm_cell/kernel
¥
;deepfake_detector/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp'deepfake_detector/lstm/lstm_cell/kernel* 
_output_shapes
:
 Ø@*
dtype0
¾
1deepfake_detector/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*B
shared_name31deepfake_detector/lstm/lstm_cell/recurrent_kernel
·
Edeepfake_detector/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp1deepfake_detector/lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
¢
%deepfake_detector/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%deepfake_detector/lstm/lstm_cell/bias

9deepfake_detector/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOp%deepfake_detector/lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
© 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ä
valueÚB× BÐ
ª
timeCnn1
avgPool1
reshape
lstm
flat
	dense
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
]
	layer
	variables
regularization_losses
trainable_variables
	keras_api
]
	layer
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
1
*0
+1
,2
-3
.4
$5
%6
 
1
*0
+1
,2
-3
.4
$5
%6
­
/non_trainable_variables

0layers
1layer_metrics
	variables
2metrics
3layer_regularization_losses
regularization_losses
	trainable_variables
 
h

*kernel
+bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api

*0
+1
 

*0
+1
­
8non_trainable_variables

9layers
:layer_metrics
	variables
;metrics
<layer_regularization_losses
regularization_losses
trainable_variables
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api
 
 
 
­
Anon_trainable_variables

Blayers
Clayer_metrics
	variables
Dmetrics
Elayer_regularization_losses
regularization_losses
trainable_variables
 
 
 
­
Fnon_trainable_variables

Glayers
Hlayer_metrics
	variables
Imetrics
Jlayer_regularization_losses
regularization_losses
trainable_variables

K
state_size

,kernel
-recurrent_kernel
.bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
 

,0
-1
.2
 

,0
-1
.2
¹
Pnon_trainable_variables

Qlayers
Rlayer_metrics
	variables
Smetrics
Tlayer_regularization_losses

Ustates
regularization_losses
trainable_variables
 
 
 
­
Vnon_trainable_variables

Wlayers
Xlayer_metrics
 	variables
Ymetrics
Zlayer_regularization_losses
!regularization_losses
"trainable_variables
[Y
VARIABLE_VALUEdeepfake_detector/dense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdeepfake_detector/dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
­
[non_trainable_variables

\layers
]layer_metrics
&	variables
^metrics
_layer_regularization_losses
'regularization_losses
(trainable_variables
ec
VARIABLE_VALUE)deepfake_detector/time_distributed/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'deepfake_detector/time_distributed/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'deepfake_detector/lstm/lstm_cell/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1deepfake_detector/lstm/lstm_cell/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%deepfake_detector/lstm/lstm_cell/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5
 
 
 

*0
+1
 

*0
+1
­
`non_trainable_variables

alayers
blayer_metrics
4	variables
cmetrics
dlayer_regularization_losses
5regularization_losses
6trainable_variables
 

0
 
 
 
 
 
 
­
enon_trainable_variables

flayers
glayer_metrics
=	variables
hmetrics
ilayer_regularization_losses
>regularization_losses
?trainable_variables
 

0
 
 
 
 
 
 
 
 
 

,0
-1
.2
 

,0
-1
.2
­
jnon_trainable_variables

klayers
llayer_metrics
L	variables
mmetrics
nlayer_regularization_losses
Mregularization_losses
Ntrainable_variables
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd*
dtype0*(
shape:ÿÿÿÿÿÿÿÿÿ<dd
Ð
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)deepfake_detector/time_distributed/kernel'deepfake_detector/time_distributed/bias'deepfake_detector/lstm/lstm_cell/kernel1deepfake_detector/lstm/lstm_cell/recurrent_kernel%deepfake_detector/lstm/lstm_cell/biasdeepfake_detector/dense/kerneldeepfake_detector/dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_7421087
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2deepfake_detector/dense/kernel/Read/ReadVariableOp0deepfake_detector/dense/bias/Read/ReadVariableOp=deepfake_detector/time_distributed/kernel/Read/ReadVariableOp;deepfake_detector/time_distributed/bias/Read/ReadVariableOp;deepfake_detector/lstm/lstm_cell/kernel/Read/ReadVariableOpEdeepfake_detector/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp9deepfake_detector/lstm/lstm_cell/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_7422951
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeepfake_detector/dense/kerneldeepfake_detector/dense/bias)deepfake_detector/time_distributed/kernel'deepfake_detector/time_distributed/bias'deepfake_detector/lstm/lstm_cell/kernel1deepfake_detector/lstm/lstm_cell/recurrent_kernel%deepfake_detector/lstm/lstm_cell/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_7422982¼î
°
`
D__inference_flatten_layer_call_and_return_conditional_losses_7422744

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Const^
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:2	
Reshape[
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs

²
&__inference_lstm_layer_call_fn_7422727

inputs
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74206272
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
Ì

'__inference_dense_layer_call_fn_7422769

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_74206542
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
Ê
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799

inputs
identity
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿcc :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
X

A__inference_lstm_layer_call_and_return_conditional_losses_7420858

inputs<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constl
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes

:2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constt
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes

:2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perms
	transpose	Transposeinputstranspose/perm:output:0*
T0*$
_output_shapes
:< Ø2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2õ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimÃ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/splitt
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm_cell/Sigmoidx
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_1y
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*
_output_shapes

:2
lstm_cell/mulk
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_1|
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm_cell/add_1x
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_2j
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterà
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7420774*
condR
while_cond_7420773*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeß
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimej
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*
_output_shapes

:2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
Í	
¬
lstm_while_cond_7421721&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_7421721___redundant_placeholder0?
;lstm_while_lstm_while_cond_7421721___redundant_placeholder1?
;lstm_while_lstm_while_cond_7421721___redundant_placeholder2?
;lstm_while_lstm_while_cond_7421721___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
Í	
¬
lstm_while_cond_7421539&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_7421539___redundant_placeholder0?
;lstm_while_lstm_while_cond_7421539___redundant_placeholder1?
;lstm_while_lstm_while_cond_7421539___redundant_placeholder2?
;lstm_while_lstm_while_cond_7421539___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
·
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
O
3__inference_average_pooling2d_layer_call_fn_7422804

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74196812
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7422085

inputs
identitys
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Reshape/shapel
ReshapeReshapeinputsReshape/shape:output:0*
T0*$
_output_shapes
:< Ø2	
Reshapea
IdentityIdentityReshape:output:0*
T0*$
_output_shapes
:< Ø2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 
 
_user_specified_nameinputs
Ú
È
while_cond_7422307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7422307___redundant_placeholder05
1while_while_cond_7422307___redundant_placeholder15
1while_while_cond_7422307___redundant_placeholder25
1while_while_cond_7422307___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ê
ü
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


(__inference_conv2d_layer_call_fn_7422789

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¶Z

A__inference_lstm_layer_call_and_return_conditional_losses_7422241
inputs_0<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2þ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp£
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp 
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimç
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422157*
condR
while_cond_7422156*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:` \
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
"
_user_specified_name
inputs/0

h
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7420475

inputs
identitys
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Reshape/shapel
ReshapeReshapeinputsReshape/shape:output:0*
T0*$
_output_shapes
:< Ø2	
Reshapea
IdentityIdentityReshape:output:0*
T0*$
_output_shapes
:< Ø2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 
 
_user_specified_nameinputs
¥\

)deepfake_detector_lstm_while_body_7419431J
Fdeepfake_detector_lstm_while_deepfake_detector_lstm_while_loop_counterP
Ldeepfake_detector_lstm_while_deepfake_detector_lstm_while_maximum_iterations,
(deepfake_detector_lstm_while_placeholder.
*deepfake_detector_lstm_while_placeholder_1.
*deepfake_detector_lstm_while_placeholder_2.
*deepfake_detector_lstm_while_placeholder_3I
Edeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1_0
deepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0[
Gdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@[
Ideepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@V
Hdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@)
%deepfake_detector_lstm_while_identity+
'deepfake_detector_lstm_while_identity_1+
'deepfake_detector_lstm_while_identity_2+
'deepfake_detector_lstm_while_identity_3+
'deepfake_detector_lstm_while_identity_4+
'deepfake_detector_lstm_while_identity_5G
Cdeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1
deepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensorY
Edeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource:
 Ø@Y
Gdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource:@T
Fdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource:@¢=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp¢>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOpñ
Ndeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2P
Ndeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape×
@deepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemdeepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0(deepfake_detector_lstm_while_placeholderWdeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype02B
@deepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpGdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02>
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp 
-deepfake_detector/lstm/while/lstm_cell/MatMulMatMulGdeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Ddeepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2/
-deepfake_detector/lstm/while/lstm_cell/MatMul
>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02@
>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp
/deepfake_detector/lstm/while/lstm_cell/MatMul_1MatMul*deepfake_detector_lstm_while_placeholder_2Fdeepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@21
/deepfake_detector/lstm/while/lstm_cell/MatMul_1þ
*deepfake_detector/lstm/while/lstm_cell/addAddV27deepfake_detector/lstm/while/lstm_cell/MatMul:product:09deepfake_detector/lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2,
*deepfake_detector/lstm/while/lstm_cell/add
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp
.deepfake_detector/lstm/while/lstm_cell/BiasAddBiasAdd.deepfake_detector/lstm/while/lstm_cell/add:z:0Edeepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@20
.deepfake_detector/lstm/while/lstm_cell/BiasAdd²
6deepfake_detector/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6deepfake_detector/lstm/while/lstm_cell/split/split_dim·
,deepfake_detector/lstm/while/lstm_cell/splitSplit?deepfake_detector/lstm/while/lstm_cell/split/split_dim:output:07deepfake_detector/lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2.
,deepfake_detector/lstm/while/lstm_cell/splitË
.deepfake_detector/lstm/while/lstm_cell/SigmoidSigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:20
.deepfake_detector/lstm/while/lstm_cell/SigmoidÏ
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_1Sigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:22
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_1ê
*deepfake_detector/lstm/while/lstm_cell/mulMul4deepfake_detector/lstm/while/lstm_cell/Sigmoid_1:y:0*deepfake_detector_lstm_while_placeholder_3*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/while/lstm_cell/mulÂ
+deepfake_detector/lstm/while/lstm_cell/ReluRelu5deepfake_detector/lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2-
+deepfake_detector/lstm/while/lstm_cell/Reluû
,deepfake_detector/lstm/while/lstm_cell/mul_1Mul2deepfake_detector/lstm/while/lstm_cell/Sigmoid:y:09deepfake_detector/lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/mul_1ð
,deepfake_detector/lstm/while/lstm_cell/add_1AddV2.deepfake_detector/lstm/while/lstm_cell/mul:z:00deepfake_detector/lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/add_1Ï
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_2Sigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:22
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_2Á
-deepfake_detector/lstm/while/lstm_cell/Relu_1Relu0deepfake_detector/lstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2/
-deepfake_detector/lstm/while/lstm_cell/Relu_1ÿ
,deepfake_detector/lstm/while/lstm_cell/mul_2Mul4deepfake_detector/lstm/while/lstm_cell/Sigmoid_2:y:0;deepfake_detector/lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/mul_2Ð
Adeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*deepfake_detector_lstm_while_placeholder_1(deepfake_detector_lstm_while_placeholder0deepfake_detector/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02C
Adeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItem
"deepfake_detector/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"deepfake_detector/lstm/while/add/yÅ
 deepfake_detector/lstm/while/addAddV2(deepfake_detector_lstm_while_placeholder+deepfake_detector/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2"
 deepfake_detector/lstm/while/add
$deepfake_detector/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$deepfake_detector/lstm/while/add_1/yé
"deepfake_detector/lstm/while/add_1AddV2Fdeepfake_detector_lstm_while_deepfake_detector_lstm_while_loop_counter-deepfake_detector/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2$
"deepfake_detector/lstm/while/add_1Ç
%deepfake_detector/lstm/while/IdentityIdentity&deepfake_detector/lstm/while/add_1:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2'
%deepfake_detector/lstm/while/Identityñ
'deepfake_detector/lstm/while/Identity_1IdentityLdeepfake_detector_lstm_while_deepfake_detector_lstm_while_maximum_iterations"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_1É
'deepfake_detector/lstm/while/Identity_2Identity$deepfake_detector/lstm/while/add:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_2ö
'deepfake_detector/lstm/while/Identity_3IdentityQdeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_3Ý
'deepfake_detector/lstm/while/Identity_4Identity0deepfake_detector/lstm/while/lstm_cell/mul_2:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/while/Identity_4Ý
'deepfake_detector/lstm/while/Identity_5Identity0deepfake_detector/lstm/while/lstm_cell/add_1:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/while/Identity_5È
!deepfake_detector/lstm/while/NoOpNoOp>^deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp=^deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp?^deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2#
!deepfake_detector/lstm/while/NoOp"
Cdeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1Edeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1_0"W
%deepfake_detector_lstm_while_identity.deepfake_detector/lstm/while/Identity:output:0"[
'deepfake_detector_lstm_while_identity_10deepfake_detector/lstm/while/Identity_1:output:0"[
'deepfake_detector_lstm_while_identity_20deepfake_detector/lstm/while/Identity_2:output:0"[
'deepfake_detector_lstm_while_identity_30deepfake_detector/lstm/while/Identity_3:output:0"[
'deepfake_detector_lstm_while_identity_40deepfake_detector/lstm/while/Identity_4:output:0"[
'deepfake_detector_lstm_while_identity_50deepfake_detector/lstm/while/Identity_5:output:0"
Fdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resourceHdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
Gdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resourceIdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"
Edeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resourceGdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0"
deepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensordeepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2~
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2|
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp2
>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 

P
4__inference_time_distributed_1_layer_call_fn_7422074

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74204652
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
á
M
1__inference_reshape_for_rnn_layer_call_fn_7422090

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:< Ø* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:< Ø2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 
 
_user_specified_nameinputs
Ú
È
while_cond_7420096
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7420096___redundant_placeholder05
1while_while_cond_7420096___redundant_placeholder15
1while_while_cond_7420096___redundant_placeholder25
1while_while_cond_7420096___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ä
Û
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815
input_1P
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
 Ø@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢%lstm/lstm_cell/BiasAdd/ReadVariableOp¢$lstm/lstm_cell/MatMul/ReadVariableOp¢&lstm/lstm_cell/MatMul_1/ReadVariableOp¢
lstm/while¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shape«
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/ReshapeÝ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DÔ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpè
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2!
time_distributed/conv2d/BiasAdd¨
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed/conv2d/Relu¡
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2"
 time_distributed/Reshape_1/shapeØ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2"
 time_distributed/Reshape_2/shape±
time_distributed/Reshape_2Reshapeinput_1)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape_2
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÍ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPool¥
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2$
"time_distributed_1/Reshape_1/shapeé
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
time_distributed_1/Reshape_1¡
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2$
"time_distributed_1/Reshape_2/shapeÓ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape_2
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape»
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:< Ø2
reshape_for_rnn/Reshapem

lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*
_output_shapes

:2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:< Ø2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
lstm/strided_slice_2¼
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMulÀ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpª
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add¹
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp«
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAdd
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim×
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter«

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_7421722*#
condR
lstm_while_cond_7421721*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeó
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¯
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm°
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddj
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:2
dense/Sigmoidc
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identityñ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
ï	
´
3__inference_deepfake_detector_layer_call_fn_7421891
input_1!
unknown: 
	unknown_0: 
	unknown_1:
 Ø@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_74209742
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
Í	
¬
lstm_while_cond_7421175&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_7421175___redundant_placeholder0?
;lstm_while_lstm_while_cond_7421175___redundant_placeholder1?
;lstm_while_lstm_while_cond_7421175___redundant_placeholder2?
;lstm_while_lstm_while_cond_7421175___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
ï

F__inference_lstm_cell_layer_call_and_return_conditional_losses_7420019

inputs

states
states_12
matmul_readvariableop_resource:
 Ø@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ë
¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7420449

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
ËB
Ô	
lstm_while_body_7421540&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0I
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@I
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@D
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorG
3lstm_while_lstm_cell_matmul_readvariableop_resource:
 Ø@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@¢+lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢*lstm/while/lstm_cell/MatMul/ReadVariableOp¢,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeê
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemÐ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpØ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulÔ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÁ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1¶
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/addÍ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpÃ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAdd
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimï
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Sigmoid
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1¢
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu³
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1¨
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1·
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2®
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5î
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
÷

F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841

inputs
states_0
states_12
matmul_readvariableop_resource:
 Ø@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ê
ü
C__inference_conv2d_layer_call_and_return_conditional_losses_7419549

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ë
¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
¶
È
while_cond_7420773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7420773___redundant_placeholder05
1while_while_cond_7420773___redundant_placeholder15
1while_while_cond_7420773___redundant_placeholder25
1while_while_cond_7420773___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
ÿE
ù
A__inference_lstm_layer_call_and_return_conditional_losses_7420166

inputs%
lstm_cell_7420084:
 Ø@#
lstm_cell_7420086:@
lstm_cell_7420088:@
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2þ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7420084lstm_cell_7420086lstm_cell_7420088*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7420084lstm_cell_7420086lstm_cell_7420088*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7420097*
condR
while_cond_7420096*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityz
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:^ Z
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs
Ê
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7419712

inputs
identity
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿcc :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
À
´
&__inference_lstm_layer_call_fn_7422705
inputs_0
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74199562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
"
_user_specified_name
inputs/0
=
´
while_body_7422157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÕ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÍ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp¶
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1«
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¸
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÿ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Relu¨
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Relu_1¬
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

E
)__inference_flatten_layer_call_fn_7422749

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_74206412
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
¸;
´
while_body_7422459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÌ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÄ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp­
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1¢
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¯
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÛ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/Relu
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1£
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
¸
ó
+__inference_lstm_cell_layer_call_fn_7422890

inputs
states_0
states_1
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ë

ó
B__inference_dense_layer_call_and_return_conditional_losses_7422760

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid]
IdentityIdentitySigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
¸
ó
+__inference_lstm_cell_layer_call_fn_7422907

inputs
states_0
states_1
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ê
Û
 __inference__traced_save_7422951
file_prefix=
9savev2_deepfake_detector_dense_kernel_read_readvariableop;
7savev2_deepfake_detector_dense_bias_read_readvariableopH
Dsavev2_deepfake_detector_time_distributed_kernel_read_readvariableopF
Bsavev2_deepfake_detector_time_distributed_bias_read_readvariableopF
Bsavev2_deepfake_detector_lstm_lstm_cell_kernel_read_readvariableopP
Lsavev2_deepfake_detector_lstm_lstm_cell_recurrent_kernel_read_readvariableopD
@savev2_deepfake_detector_lstm_lstm_cell_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¹
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ë
valueÁB¾B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_deepfake_detector_dense_kernel_read_readvariableop7savev2_deepfake_detector_dense_bias_read_readvariableopDsavev2_deepfake_detector_time_distributed_kernel_read_readvariableopBsavev2_deepfake_detector_time_distributed_bias_read_readvariableopBsavev2_deepfake_detector_lstm_lstm_cell_kernel_read_readvariableopLsavev2_deepfake_detector_lstm_lstm_cell_recurrent_kernel_read_readvariableop@savev2_deepfake_detector_lstm_lstm_cell_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*[
_input_shapesJ
H: ::: : :
 Ø@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
 Ø@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: 

¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2
	Reshape_1
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
µ	
¦
%__inference_signature_wrapper_7421087
input_1!
unknown: 
	unknown_0: 
	unknown_1:
 Ø@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_74195242
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
ª
P
4__inference_time_distributed_1_layer_call_fn_7422069

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74197542
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
î
Õ
M__inference_time_distributed_layer_call_and_return_conditional_losses_7419612

inputs(
conv2d_7419600: 
conv2d_7419602: 
identity¢conv2d/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshape¢
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_7419600conv2d_7419602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape«
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2
	Reshape_1
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identityo
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ì	
³
3__inference_deepfake_detector_layer_call_fn_7421853

inputs!
unknown: 
	unknown_0: 
	unknown_1:
 Ø@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_74206612
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
ËB
Ô	
lstm_while_body_7421176&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0I
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@I
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@D
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorG
3lstm_while_lstm_cell_matmul_readvariableop_resource:
 Ø@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@¢+lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢*lstm/while/lstm_cell/MatMul/ReadVariableOp¢,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeê
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemÐ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpØ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulÔ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÁ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1¶
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/addÍ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpÃ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAdd
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimï
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Sigmoid
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1¢
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu³
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1¨
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1·
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2®
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5î
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
Ú
È
while_cond_7422156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7422156___redundant_placeholder05
1while_while_cond_7422156___redundant_placeholder15
1while_while_cond_7422156___redundant_placeholder25
1while_while_cond_7422156___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
÷

F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873

inputs
states_0
states_12
matmul_readvariableop_resource:
 Ø@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¶Z

A__inference_lstm_layer_call_and_return_conditional_losses_7422392
inputs_0<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2þ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp£
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp 
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimç
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422308*
condR
while_cond_7422307*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:` \
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
"
_user_specified_name
inputs/0
Ë
¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7420918

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2
Reshape_1/shape
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
Ë

ó
B__inference_dense_layer_call_and_return_conditional_losses_7420654

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid]
IdentityIdentitySigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
ã
§
2__inference_time_distributed_layer_call_fn_7421987

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74196122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs

P
4__inference_time_distributed_1_layer_call_fn_7422079

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74208892
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
¶
È
while_cond_7422609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7422609___redundant_placeholder05
1while_while_cond_7422609___redundant_placeholder15
1while_while_cond_7422609___redundant_placeholder25
1while_while_cond_7422609___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
À
Ú
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
 Ø@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢%lstm/lstm_cell/BiasAdd/ReadVariableOp¢$lstm/lstm_cell/MatMul/ReadVariableOp¢&lstm/lstm_cell/MatMul_1/ReadVariableOp¢
lstm/while¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shapeª
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/ReshapeÝ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DÔ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpè
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2!
time_distributed/conv2d/BiasAdd¨
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed/conv2d/Relu¡
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2"
 time_distributed/Reshape_1/shapeØ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2"
 time_distributed/Reshape_2/shape°
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape_2
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÍ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPool¥
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2$
"time_distributed_1/Reshape_1/shapeé
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
time_distributed_1/Reshape_1¡
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2$
"time_distributed_1/Reshape_2/shapeÓ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape_2
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape»
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:< Ø2
reshape_for_rnn/Reshapem

lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*
_output_shapes

:2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:< Ø2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
lstm/strided_slice_2¼
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMulÀ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpª
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add¹
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp«
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAdd
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim×
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter«

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_7421176*#
condR
lstm_while_cond_7421175*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeó
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¯
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm°
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddj
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:2
dense/Sigmoidc
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identityñ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
¶
È
while_cond_7420542
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7420542___redundant_placeholder05
1while_while_cond_7420542___redundant_placeholder15
1while_while_cond_7420542___redundant_placeholder25
1while_while_cond_7420542___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
¡%
Ó
while_body_7419887
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_7419911_0:
 Ø@+
while_lstm_cell_7419913_0:@'
while_lstm_cell_7419915_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_7419911:
 Ø@)
while_lstm_cell_7419913:@%
while_lstm_cell_7419915:@¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÕ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÖ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7419911_0while_lstm_cell_7419913_0while_lstm_cell_7419915_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¡
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¡
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_7419911while_lstm_cell_7419911_0"4
while_lstm_cell_7419913while_lstm_cell_7419913_0"4
while_lstm_cell_7419915while_lstm_cell_7419915_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
X

A__inference_lstm_layer_call_and_return_conditional_losses_7422543

inputs<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constl
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes

:2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constt
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes

:2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perms
	transpose	Transposeinputstranspose/perm:output:0*
T0*$
_output_shapes
:< Ø2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2õ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimÃ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/splitt
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm_cell/Sigmoidx
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_1y
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*
_output_shapes

:2
lstm_cell/mulk
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_1|
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm_cell/add_1x
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_2j
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterà
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422459*
condR
while_cond_7422458*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeß
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimej
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*
_output_shapes

:2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
ï	
´
3__inference_deepfake_detector_layer_call_fn_7421834
input_1!
unknown: 
	unknown_0: 
	unknown_1:
 Ø@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_74206612
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
Í	
¬
lstm_while_cond_7421357&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_7421357___redundant_placeholder0?
;lstm_while_lstm_while_cond_7421357___redundant_placeholder1?
;lstm_while_lstm_while_cond_7421357___redundant_placeholder2?
;lstm_while_lstm_while_cond_7421357___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
À
Ú
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
 Ø@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢%lstm/lstm_cell/BiasAdd/ReadVariableOp¢$lstm/lstm_cell/MatMul/ReadVariableOp¢&lstm/lstm_cell/MatMul_1/ReadVariableOp¢
lstm/while¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shapeª
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/ReshapeÝ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DÔ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpè
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2!
time_distributed/conv2d/BiasAdd¨
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed/conv2d/Relu¡
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2"
 time_distributed/Reshape_1/shapeØ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2"
 time_distributed/Reshape_2/shape°
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape_2
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÍ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPool¥
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2$
"time_distributed_1/Reshape_1/shapeé
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
time_distributed_1/Reshape_1¡
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2$
"time_distributed_1/Reshape_2/shapeÓ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape_2
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape»
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:< Ø2
reshape_for_rnn/Reshapem

lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*
_output_shapes

:2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:< Ø2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
lstm/strided_slice_2¼
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMulÀ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpª
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add¹
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp«
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAdd
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim×
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter«

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_7421358*#
condR
lstm_while_cond_7421357*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeó
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¯
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm°
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddj
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:2
dense/Sigmoidc
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identityñ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
Ä
Û
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633
input_1P
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
 Ø@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢%lstm/lstm_cell/BiasAdd/ReadVariableOp¢$lstm/lstm_cell/MatMul/ReadVariableOp¢&lstm/lstm_cell/MatMul_1/ReadVariableOp¢
lstm/while¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shape«
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/ReshapeÝ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DÔ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpè
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2!
time_distributed/conv2d/BiasAdd¨
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed/conv2d/Relu¡
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2"
 time_distributed/Reshape_1/shapeØ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2"
 time_distributed/Reshape_2/shape±
time_distributed/Reshape_2Reshapeinput_1)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape_2
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÍ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPool¥
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2$
"time_distributed_1/Reshape_1/shapeé
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
time_distributed_1/Reshape_1¡
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2$
"time_distributed_1/Reshape_2/shapeÓ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape_2
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape»
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:< Ø2
reshape_for_rnn/Reshapem

lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*
_output_shapes

:2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:< Ø2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
lstm/strided_slice_2¼
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp®
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMulÀ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpª
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add¹
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp«
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAdd
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim×
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter«

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_7421540*#
condR
lstm_while_cond_7421539*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeó
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¯
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm°
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddj
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:2
dense/Sigmoidc
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identityñ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
¿
§
2__inference_time_distributed_layer_call_fn_7421996

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74204492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
î
Õ
M__inference_time_distributed_layer_call_and_return_conditional_losses_7419562

inputs(
conv2d_7419550: 
conv2d_7419552: 
identity¢conv2d/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshape¢
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_7419550conv2d_7419552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape«
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2
	Reshape_1
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identityo
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ì	
³
3__inference_deepfake_detector_layer_call_fn_7421872

inputs!
unknown: 
	unknown_0: 
	unknown_1:
 Ø@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_74209742
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
·
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7419681

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
È
while_cond_7419886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7419886___redundant_placeholder05
1while_while_cond_7419886___redundant_placeholder15
1while_while_cond_7419886___redundant_placeholder25
1while_while_cond_7419886___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¡%
Ó
while_body_7420097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_7420121_0:
 Ø@+
while_lstm_cell_7420123_0:@'
while_lstm_cell_7420125_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_7420121:
 Ø@)
while_lstm_cell_7420123:@%
while_lstm_cell_7420125:@¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÕ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÖ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7420121_0while_lstm_cell_7420123_0while_lstm_cell_7420125_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¡
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¡
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_7420121while_lstm_cell_7420121_0"4
while_lstm_cell_7420123while_lstm_cell_7420123_0"4
while_lstm_cell_7420125while_lstm_cell_7420125_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
%
î
#__inference__traced_restore_7422982
file_prefixA
/assignvariableop_deepfake_detector_dense_kernel:=
/assignvariableop_1_deepfake_detector_dense_bias:V
<assignvariableop_2_deepfake_detector_time_distributed_kernel: H
:assignvariableop_3_deepfake_detector_time_distributed_bias: N
:assignvariableop_4_deepfake_detector_lstm_lstm_cell_kernel:
 Ø@V
Dassignvariableop_5_deepfake_detector_lstm_lstm_cell_recurrent_kernel:@F
8assignvariableop_6_deepfake_detector_lstm_lstm_cell_bias:@

identity_8¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¿
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ë
valueÁB¾B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slicesÓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity®
AssignVariableOpAssignVariableOp/assignvariableop_deepfake_detector_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1´
AssignVariableOp_1AssignVariableOp/assignvariableop_1_deepfake_detector_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Á
AssignVariableOp_2AssignVariableOp<assignvariableop_2_deepfake_detector_time_distributed_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¿
AssignVariableOp_3AssignVariableOp:assignvariableop_3_deepfake_detector_time_distributed_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¿
AssignVariableOp_4AssignVariableOp:assignvariableop_4_deepfake_detector_lstm_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5É
AssignVariableOp_5AssignVariableOpDassignvariableop_5_deepfake_detector_lstm_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_deepfake_detector_lstm_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpù

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7c

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_8ã
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¿
§
2__inference_time_distributed_layer_call_fn_7422005

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74209182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
¶
È
while_cond_7422458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_7422458___redundant_placeholder05
1while_while_cond_7422458___redundant_placeholder15
1while_while_cond_7422458___redundant_placeholder25
1while_while_cond_7422458___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2
Reshape_1/shape
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
ËB
Ô	
lstm_while_body_7421722&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0I
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@I
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@D
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorG
3lstm_while_lstm_cell_matmul_readvariableop_resource:
 Ø@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@¢+lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢*lstm/while/lstm_cell/MatMul/ReadVariableOp¢,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeê
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemÐ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpØ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulÔ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÁ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1¶
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/addÍ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpÃ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAdd
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimï
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Sigmoid
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1¢
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu³
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1¨
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1·
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2®
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5î
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
X

A__inference_lstm_layer_call_and_return_conditional_losses_7420627

inputs<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constl
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes

:2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constt
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes

:2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perms
	transpose	Transposeinputstranspose/perm:output:0*
T0*$
_output_shapes
:< Ø2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2õ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimÃ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/splitt
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm_cell/Sigmoidx
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_1y
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*
_output_shapes

:2
lstm_cell/mulk
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_1|
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm_cell/add_1x
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_2j
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterà
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7420543*
condR
while_cond_7420542*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeß
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimej
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*
_output_shapes

:2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
ª
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7419721

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
Reshape
!average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122#
!average_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape®
	Reshape_1Reshape*average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
Ü"
Ï
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7420974

inputs2
time_distributed_7420949: &
time_distributed_7420951:  
lstm_7420960:
 Ø@
lstm_7420962:@
lstm_7420964:@
dense_7420968:
dense_7420970:
identity¢dense/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallÎ
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_7420949time_distributed_7420951*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74209182*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shapeª
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape­
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74208892$
"time_distributed_1/PartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÛ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
reshape_for_rnn/PartitionedCallPartitionedCall+time_distributed_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:< Ø* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752!
reshape_for_rnn/PartitionedCall¯
lstm/StatefulPartitionedCallStatefulPartitionedCall(reshape_for_rnn/PartitionedCall:output:0lstm_7420960lstm_7420962lstm_7420964*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74208582
lstm/StatefulPartitionedCallë
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_74206412
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7420968dense_7420970*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_74206542
dense/StatefulPartitionedCallx
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity¸
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
ã
§
2__inference_time_distributed_layer_call_fn_7421978

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74195622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
X

A__inference_lstm_layer_call_and_return_conditional_losses_7422694

inputs<
(lstm_cell_matmul_readvariableop_resource:
 Ø@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identity¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constl
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes

:2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constt
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes

:2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perms
	transpose	Transposeinputstranspose/perm:output:0*
T0*$
_output_shapes
:< Ø2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2õ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2
strided_slice_2­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul±
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addª
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimÃ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/splitt
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm_cell/Sigmoidx
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_1y
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*
_output_shapes

:2
lstm_cell/mulk
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_1|
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm_cell/add_1x
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm_cell/Sigmoid_2j
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterà
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422610*
condR
while_cond_7422609*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeß
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:<2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimej
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*
_output_shapes

:2

Identity¿
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
¸;
´
while_body_7420543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÌ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÄ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp­
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1¢
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¯
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÛ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/Relu
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1£
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
ÿE
ù
A__inference_lstm_layer_call_and_return_conditional_losses_7419956

inputs%
lstm_cell_7419874:
 Ø@#
lstm_cell_7419876:@
lstm_cell_7419878:@
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2þ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7419874lstm_cell_7419876lstm_cell_7419878*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7419874lstm_cell_7419876lstm_cell_7419878*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7419887*
condR
while_cond_7419886*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityz
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:^ Z
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs
¸;
´
while_body_7422610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÌ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÄ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp­
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1¢
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¯
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÛ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/Relu
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1£
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
ó
O
3__inference_average_pooling2d_layer_call_fn_7422809

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿcc :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
Ë
¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ<dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs
Ü"
Ï
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7420661

inputs2
time_distributed_7420450: &
time_distributed_7420452:  
lstm_7420628:
 Ø@
lstm_7420630:@
lstm_7420632:@
dense_7420655:
dense_7420657:
identity¢dense/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallÎ
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_7420450time_distributed_7420452*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74204492*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2 
time_distributed/Reshape/shapeª
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2
time_distributed/Reshape­
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74204652$
"time_distributed_1/PartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2"
 time_distributed_1/Reshape/shapeÛ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
time_distributed_1/Reshape
reshape_for_rnn/PartitionedCallPartitionedCall+time_distributed_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:< Ø* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752!
reshape_for_rnn/PartitionedCall¯
lstm/StatefulPartitionedCallStatefulPartitionedCall(reshape_for_rnn/PartitionedCall:output:0lstm_7420628lstm_7420630lstm_7420632*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74206272
lstm/StatefulPartitionedCallë
flatten/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_74206412
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7420655dense_7420657*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_74206542
dense/StatefulPartitionedCallx
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity¸
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
 
_user_specified_nameinputs

²
&__inference_lstm_layer_call_fn_7422738

inputs
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74208582
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:< Ø: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:< Ø
 
_user_specified_nameinputs
ËB
Ô	
lstm_while_body_7421358&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0I
5lstm_while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@I
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@D
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorG
3lstm_while_lstm_cell_matmul_readvariableop_resource:
 Ø@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@¢+lstm/while/lstm_cell/BiasAdd/ReadVariableOp¢*lstm/while/lstm_cell/MatMul/ReadVariableOp¢,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeê
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemÐ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpØ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulÔ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpÁ
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1¶
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/addÍ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpÃ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAdd
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimï
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Sigmoid
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1¢
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu³
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1¨
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1·
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2®
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5î
lstm/while/NoOpNoOp,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm/while/NoOp"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 

¢
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshapeª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÃ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :c2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2
	Reshape_1
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 2

Identity
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
=
´
while_body_7422308
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÕ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÍ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp¶
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1«
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¸
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÿ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Relu¨
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/Relu_1¬
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7420889

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2
Reshape_1/shape
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
ï

F__inference_lstm_cell_layer_call_and_return_conditional_losses_7419873

inputs

states
states_12
matmul_readvariableop_resource:
 Ø@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ Ø:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7420465

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       2
Reshape_1/shape
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc :[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 
 
_user_specified_nameinputs
ê

)deepfake_detector_lstm_while_cond_7419430J
Fdeepfake_detector_lstm_while_deepfake_detector_lstm_while_loop_counterP
Ldeepfake_detector_lstm_while_deepfake_detector_lstm_while_maximum_iterations,
(deepfake_detector_lstm_while_placeholder.
*deepfake_detector_lstm_while_placeholder_1.
*deepfake_detector_lstm_while_placeholder_2.
*deepfake_detector_lstm_while_placeholder_3L
Hdeepfake_detector_lstm_while_less_deepfake_detector_lstm_strided_slice_1c
_deepfake_detector_lstm_while_deepfake_detector_lstm_while_cond_7419430___redundant_placeholder0c
_deepfake_detector_lstm_while_deepfake_detector_lstm_while_cond_7419430___redundant_placeholder1c
_deepfake_detector_lstm_while_deepfake_detector_lstm_while_cond_7419430___redundant_placeholder2c
_deepfake_detector_lstm_while_deepfake_detector_lstm_while_cond_7419430___redundant_placeholder3)
%deepfake_detector_lstm_while_identity
ã
!deepfake_detector/lstm/while/LessLess(deepfake_detector_lstm_while_placeholderHdeepfake_detector_lstm_while_less_deepfake_detector_lstm_strided_slice_1*
T0*
_output_shapes
: 2#
!deepfake_detector/lstm/while/Less¢
%deepfake_detector/lstm/while/IdentityIdentity%deepfake_detector/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2'
%deepfake_detector/lstm/while/Identity"W
%deepfake_detector_lstm_while_identity.deepfake_detector/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
:
À
´
&__inference_lstm_layer_call_fn_7422716
inputs_0
unknown:
 Ø@
	unknown_0:@
	unknown_1:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74201662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø: : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
": ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø
"
_user_specified_name
inputs/0
æ
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422041

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¦
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
°
`
D__inference_flatten_layer_call_and_return_conditional_losses_7420641

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Const^
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

:2	
Reshape[
IdentityIdentityReshape:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
ª
P
4__inference_time_distributed_1_layer_call_fn_7422064

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74197212
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
¸;
´
while_body_7420774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
0while_lstm_cell_matmul_readvariableop_resource_0:
 Ø@D
2while_lstm_cell_matmul_1_readvariableop_resource_0:@?
1while_lstm_cell_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
.while_lstm_cell_matmul_readvariableop_resource:
 Ø@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÌ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
 Ø*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÁ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
 Ø@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpÄ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMulÅ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp­
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1¢
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/add¾
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp¯
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAdd
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimÛ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/Relu
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1£
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Õ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: 
Ý¿
½
"__inference__wrapped_model_7419524
input_1b
Hdeepfake_detector_time_distributed_conv2d_conv2d_readvariableop_resource: W
Ideepfake_detector_time_distributed_conv2d_biasadd_readvariableop_resource: S
?deepfake_detector_lstm_lstm_cell_matmul_readvariableop_resource:
 Ø@S
Adeepfake_detector_lstm_lstm_cell_matmul_1_readvariableop_resource:@N
@deepfake_detector_lstm_lstm_cell_biasadd_readvariableop_resource:@H
6deepfake_detector_dense_matmul_readvariableop_resource:E
7deepfake_detector_dense_biasadd_readvariableop_resource:
identity¢.deepfake_detector/dense/BiasAdd/ReadVariableOp¢-deepfake_detector/dense/MatMul/ReadVariableOp¢7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp¢6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp¢8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp¢deepfake_detector/lstm/while¢@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp¢?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp½
0deepfake_detector/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      22
0deepfake_detector/time_distributed/Reshape/shapeá
*deepfake_detector/time_distributed/ReshapeReshapeinput_19deepfake_detector/time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2,
*deepfake_detector/time_distributed/Reshape
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOpHdeepfake_detector_time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02A
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOpÏ
0deepfake_detector/time_distributed/conv2d/Conv2DConv2D3deepfake_detector/time_distributed/Reshape:output:0Gdeepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc *
paddingVALID*
strides
22
0deepfake_detector/time_distributed/conv2d/Conv2D
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOpIdeepfake_detector_time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp°
1deepfake_detector/time_distributed/conv2d/BiasAddBiasAdd9deepfake_detector/time_distributed/conv2d/Conv2D:output:0Hdeepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 23
1deepfake_detector/time_distributed/conv2d/BiasAddÞ
.deepfake_detector/time_distributed/conv2d/ReluRelu:deepfake_detector/time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 20
.deepfake_detector/time_distributed/conv2d/ReluÅ
2deepfake_detector/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   c   c       24
2deepfake_detector/time_distributed/Reshape_1/shape 
,deepfake_detector/time_distributed/Reshape_1Reshape<deepfake_detector/time_distributed/conv2d/Relu:activations:0;deepfake_detector/time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<cc 2.
,deepfake_detector/time_distributed/Reshape_1Á
2deepfake_detector/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      24
2deepfake_detector/time_distributed/Reshape_2/shapeç
,deepfake_detector/time_distributed/Reshape_2Reshapeinput_1;deepfake_detector/time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2.
,deepfake_detector/time_distributed/Reshape_2Á
2deepfake_detector/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       24
2deepfake_detector/time_distributed_1/Reshape/shape
,deepfake_detector/time_distributed_1/ReshapeReshape5deepfake_detector/time_distributed/Reshape_1:output:0;deepfake_detector/time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2.
,deepfake_detector/time_distributed_1/Reshape¸
>deepfake_detector/time_distributed_1/average_pooling2d/AvgPoolAvgPool5deepfake_detector/time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2@
>deepfake_detector/time_distributed_1/average_pooling2d/AvgPoolÉ
4deepfake_detector/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ<   1   1       26
4deepfake_detector/time_distributed_1/Reshape_1/shape±
.deepfake_detector/time_distributed_1/Reshape_1ReshapeGdeepfake_detector/time_distributed_1/average_pooling2d/AvgPool:output:0=deepfake_detector/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<11 20
.deepfake_detector/time_distributed_1/Reshape_1Å
4deepfake_detector/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       26
4deepfake_detector/time_distributed_1/Reshape_2/shape
.deepfake_detector/time_distributed_1/Reshape_2Reshape5deepfake_detector/time_distributed/Reshape_1:output:0=deepfake_detector/time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 20
.deepfake_detector/time_distributed_1/Reshape_2·
/deepfake_detector/reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 21
/deepfake_detector/reshape_for_rnn/Reshape/shape
)deepfake_detector/reshape_for_rnn/ReshapeReshape7deepfake_detector/time_distributed_1/Reshape_1:output:08deepfake_detector/reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:< Ø2+
)deepfake_detector/reshape_for_rnn/Reshape
deepfake_detector/lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
deepfake_detector/lstm/Shape¢
*deepfake_detector/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*deepfake_detector/lstm/strided_slice/stack¦
,deepfake_detector/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,deepfake_detector/lstm/strided_slice/stack_1¦
,deepfake_detector/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,deepfake_detector/lstm/strided_slice/stack_2ì
$deepfake_detector/lstm/strided_sliceStridedSlice%deepfake_detector/lstm/Shape:output:03deepfake_detector/lstm/strided_slice/stack:output:05deepfake_detector/lstm/strided_slice/stack_1:output:05deepfake_detector/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$deepfake_detector/lstm/strided_slice
"deepfake_detector/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"deepfake_detector/lstm/zeros/mul/yÈ
 deepfake_detector/lstm/zeros/mulMul-deepfake_detector/lstm/strided_slice:output:0+deepfake_detector/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2"
 deepfake_detector/lstm/zeros/mul
#deepfake_detector/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#deepfake_detector/lstm/zeros/Less/yÃ
!deepfake_detector/lstm/zeros/LessLess$deepfake_detector/lstm/zeros/mul:z:0,deepfake_detector/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2#
!deepfake_detector/lstm/zeros/Less
%deepfake_detector/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%deepfake_detector/lstm/zeros/packed/1ß
#deepfake_detector/lstm/zeros/packedPack-deepfake_detector/lstm/strided_slice:output:0.deepfake_detector/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#deepfake_detector/lstm/zeros/packed
"deepfake_detector/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"deepfake_detector/lstm/zeros/ConstÈ
deepfake_detector/lstm/zerosFill,deepfake_detector/lstm/zeros/packed:output:0+deepfake_detector/lstm/zeros/Const:output:0*
T0*
_output_shapes

:2
deepfake_detector/lstm/zeros
$deepfake_detector/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$deepfake_detector/lstm/zeros_1/mul/yÎ
"deepfake_detector/lstm/zeros_1/mulMul-deepfake_detector/lstm/strided_slice:output:0-deepfake_detector/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2$
"deepfake_detector/lstm/zeros_1/mul
%deepfake_detector/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2'
%deepfake_detector/lstm/zeros_1/Less/yË
#deepfake_detector/lstm/zeros_1/LessLess&deepfake_detector/lstm/zeros_1/mul:z:0.deepfake_detector/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2%
#deepfake_detector/lstm/zeros_1/Less
'deepfake_detector/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'deepfake_detector/lstm/zeros_1/packed/1å
%deepfake_detector/lstm/zeros_1/packedPack-deepfake_detector/lstm/strided_slice:output:00deepfake_detector/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%deepfake_detector/lstm/zeros_1/packed
$deepfake_detector/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$deepfake_detector/lstm/zeros_1/ConstÐ
deepfake_detector/lstm/zeros_1Fill.deepfake_detector/lstm/zeros_1/packed:output:0-deepfake_detector/lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2 
deepfake_detector/lstm/zeros_1£
%deepfake_detector/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%deepfake_detector/lstm/transpose/permä
 deepfake_detector/lstm/transpose	Transpose2deepfake_detector/reshape_for_rnn/Reshape:output:0.deepfake_detector/lstm/transpose/perm:output:0*
T0*$
_output_shapes
:< Ø2"
 deepfake_detector/lstm/transpose
deepfake_detector/lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2 
deepfake_detector/lstm/Shape_1¦
,deepfake_detector/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,deepfake_detector/lstm/strided_slice_1/stackª
.deepfake_detector/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_1/stack_1ª
.deepfake_detector/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_1/stack_2ø
&deepfake_detector/lstm/strided_slice_1StridedSlice'deepfake_detector/lstm/Shape_1:output:05deepfake_detector/lstm/strided_slice_1/stack:output:07deepfake_detector/lstm/strided_slice_1/stack_1:output:07deepfake_detector/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_1³
2deepfake_detector/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2deepfake_detector/lstm/TensorArrayV2/element_shape
$deepfake_detector/lstm/TensorArrayV2TensorListReserve;deepfake_detector/lstm/TensorArrayV2/element_shape:output:0/deepfake_detector/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$deepfake_detector/lstm/TensorArrayV2í
Ldeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2N
Ldeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeÔ
>deepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$deepfake_detector/lstm/transpose:y:0Udeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>deepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor¦
,deepfake_detector/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,deepfake_detector/lstm/strided_slice_2/stackª
.deepfake_detector/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_2/stack_1ª
.deepfake_detector/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_2/stack_2ÿ
&deepfake_detector/lstm/strided_slice_2StridedSlice$deepfake_detector/lstm/transpose:y:05deepfake_detector/lstm/strided_slice_2/stack:output:07deepfake_detector/lstm/strided_slice_2/stack_1:output:07deepfake_detector/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
 Ø*
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_2ò
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp?deepfake_detector_lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
 Ø@*
dtype028
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOpö
'deepfake_detector/lstm/lstm_cell/MatMulMatMul/deepfake_detector/lstm/strided_slice_2:output:0>deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2)
'deepfake_detector/lstm/lstm_cell/MatMulö
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAdeepfake_detector_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02:
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOpò
)deepfake_detector/lstm/lstm_cell/MatMul_1MatMul%deepfake_detector/lstm/zeros:output:0@deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2+
)deepfake_detector/lstm/lstm_cell/MatMul_1æ
$deepfake_detector/lstm/lstm_cell/addAddV21deepfake_detector/lstm/lstm_cell/MatMul:product:03deepfake_detector/lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2&
$deepfake_detector/lstm/lstm_cell/addï
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp@deepfake_detector_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOpó
(deepfake_detector/lstm/lstm_cell/BiasAddBiasAdd(deepfake_detector/lstm/lstm_cell/add:z:0?deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2*
(deepfake_detector/lstm/lstm_cell/BiasAdd¦
0deepfake_detector/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0deepfake_detector/lstm/lstm_cell/split/split_dim
&deepfake_detector/lstm/lstm_cell/splitSplit9deepfake_detector/lstm/lstm_cell/split/split_dim:output:01deepfake_detector/lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2(
&deepfake_detector/lstm/lstm_cell/split¹
(deepfake_detector/lstm/lstm_cell/SigmoidSigmoid/deepfake_detector/lstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2*
(deepfake_detector/lstm/lstm_cell/Sigmoid½
*deepfake_detector/lstm/lstm_cell/Sigmoid_1Sigmoid/deepfake_detector/lstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/lstm_cell/Sigmoid_1Õ
$deepfake_detector/lstm/lstm_cell/mulMul.deepfake_detector/lstm/lstm_cell/Sigmoid_1:y:0'deepfake_detector/lstm/zeros_1:output:0*
T0*
_output_shapes

:2&
$deepfake_detector/lstm/lstm_cell/mul°
%deepfake_detector/lstm/lstm_cell/ReluRelu/deepfake_detector/lstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2'
%deepfake_detector/lstm/lstm_cell/Reluã
&deepfake_detector/lstm/lstm_cell/mul_1Mul,deepfake_detector/lstm/lstm_cell/Sigmoid:y:03deepfake_detector/lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/mul_1Ø
&deepfake_detector/lstm/lstm_cell/add_1AddV2(deepfake_detector/lstm/lstm_cell/mul:z:0*deepfake_detector/lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/add_1½
*deepfake_detector/lstm/lstm_cell/Sigmoid_2Sigmoid/deepfake_detector/lstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/lstm_cell/Sigmoid_2¯
'deepfake_detector/lstm/lstm_cell/Relu_1Relu*deepfake_detector/lstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/lstm_cell/Relu_1ç
&deepfake_detector/lstm/lstm_cell/mul_2Mul.deepfake_detector/lstm/lstm_cell/Sigmoid_2:y:05deepfake_detector/lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/mul_2½
4deepfake_detector/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      26
4deepfake_detector/lstm/TensorArrayV2_1/element_shape
&deepfake_detector/lstm/TensorArrayV2_1TensorListReserve=deepfake_detector/lstm/TensorArrayV2_1/element_shape:output:0/deepfake_detector/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&deepfake_detector/lstm/TensorArrayV2_1|
deepfake_detector/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
deepfake_detector/lstm/time­
/deepfake_detector/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/deepfake_detector/lstm/while/maximum_iterations
)deepfake_detector/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2+
)deepfake_detector/lstm/while/loop_counter¹
deepfake_detector/lstm/whileWhile2deepfake_detector/lstm/while/loop_counter:output:08deepfake_detector/lstm/while/maximum_iterations:output:0$deepfake_detector/lstm/time:output:0/deepfake_detector/lstm/TensorArrayV2_1:handle:0%deepfake_detector/lstm/zeros:output:0'deepfake_detector/lstm/zeros_1:output:0/deepfake_detector/lstm/strided_slice_1:output:0Ndeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0?deepfake_detector_lstm_lstm_cell_matmul_readvariableop_resourceAdeepfake_detector_lstm_lstm_cell_matmul_1_readvariableop_resource@deepfake_detector_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)deepfake_detector_lstm_while_body_7419431*5
cond-R+
)deepfake_detector_lstm_while_cond_7419430*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations 2
deepfake_detector/lstm/whileã
Gdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shape»
9deepfake_detector/lstm/TensorArrayV2Stack/TensorListStackTensorListStack%deepfake_detector/lstm/while:output:3Pdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02;
9deepfake_detector/lstm/TensorArrayV2Stack/TensorListStack¯
,deepfake_detector/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2.
,deepfake_detector/lstm/strided_slice_3/stackª
.deepfake_detector/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.deepfake_detector/lstm/strided_slice_3/stack_1ª
.deepfake_detector/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_3/stack_2
&deepfake_detector/lstm/strided_slice_3StridedSliceBdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack:tensor:05deepfake_detector/lstm/strided_slice_3/stack:output:07deepfake_detector/lstm/strided_slice_3/stack_1:output:07deepfake_detector/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_3§
'deepfake_detector/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'deepfake_detector/lstm/transpose_1/permø
"deepfake_detector/lstm/transpose_1	TransposeBdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack:tensor:00deepfake_detector/lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2$
"deepfake_detector/lstm/transpose_1
deepfake_detector/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2 
deepfake_detector/lstm/runtime
deepfake_detector/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2!
deepfake_detector/flatten/ConstÕ
!deepfake_detector/flatten/ReshapeReshape/deepfake_detector/lstm/strided_slice_3:output:0(deepfake_detector/flatten/Const:output:0*
T0*
_output_shapes

:2#
!deepfake_detector/flatten/ReshapeÕ
-deepfake_detector/dense/MatMul/ReadVariableOpReadVariableOp6deepfake_detector_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-deepfake_detector/dense/MatMul/ReadVariableOpÖ
deepfake_detector/dense/MatMulMatMul*deepfake_detector/flatten/Reshape:output:05deepfake_detector/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 
deepfake_detector/dense/MatMulÔ
.deepfake_detector/dense/BiasAdd/ReadVariableOpReadVariableOp7deepfake_detector_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.deepfake_detector/dense/BiasAdd/ReadVariableOpØ
deepfake_detector/dense/BiasAddBiasAdd(deepfake_detector/dense/MatMul:product:06deepfake_detector/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2!
deepfake_detector/dense/BiasAdd 
deepfake_detector/dense/SigmoidSigmoid(deepfake_detector/dense/BiasAdd:output:0*
T0*
_output_shapes

:2!
deepfake_detector/dense/Sigmoidu
IdentityIdentity#deepfake_detector/dense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp/^deepfake_detector/dense/BiasAdd/ReadVariableOp.^deepfake_detector/dense/MatMul/ReadVariableOp8^deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp7^deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp9^deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp^deepfake_detector/lstm/whileA^deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp@^deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ<dd: : : : : : : 2`
.deepfake_detector/dense/BiasAdd/ReadVariableOp.deepfake_detector/dense/BiasAdd/ReadVariableOp2^
-deepfake_detector/dense/MatMul/ReadVariableOp-deepfake_detector/dense/MatMul/ReadVariableOp2r
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp2p
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp2t
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp2<
deepfake_detector/lstm/whiledeepfake_detector/lstm/while2
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp2
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ<dd
!
_user_specified_name	input_1
ª
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7419754

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
Reshape
!average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122#
!average_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape®
	Reshape_1Reshape*average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs
æ
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422023

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿc   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿcc 2	
ReshapeÉ
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11 *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :12
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4à
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¦
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
G
input_1<
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<dd3
output_1'
StatefulPartitionedCall:0tensorflow/serving/predict:ÿÃ

timeCnn1
avgPool1
reshape
lstm
flat
	dense
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
o_default_save_signature
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_model
°
	layer
	variables
regularization_losses
trainable_variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_layer
°
	layer
	variables
regularization_losses
trainable_variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
¥
	variables
regularization_losses
trainable_variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"
_tf_keras_layer
Ã
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"
_tf_keras_rnn_layer
¥
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*z&call_and_return_all_conditional_losses
{__call__"
_tf_keras_layer
»

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
*|&call_and_return_all_conditional_losses
}__call__"
_tf_keras_layer
Q
*0
+1
,2
-3
.4
$5
%6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
*0
+1
,2
-3
.4
$5
%6"
trackable_list_wrapper
Ê
/non_trainable_variables

0layers
1layer_metrics
	variables
2metrics
3layer_regularization_losses
regularization_losses
	trainable_variables
q__call__
o_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
,
~serving_default"
signature_map
¼

*kernel
+bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
*&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
­
8non_trainable_variables

9layers
:layer_metrics
	variables
;metrics
<layer_regularization_losses
regularization_losses
trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
§
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Clayer_metrics
	variables
Dmetrics
Elayer_regularization_losses
regularization_losses
trainable_variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hlayer_metrics
	variables
Imetrics
Jlayer_regularization_losses
regularization_losses
trainable_variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ã
K
state_size

,kernel
-recurrent_kernel
.bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
¹
Pnon_trainable_variables

Qlayers
Rlayer_metrics
	variables
Smetrics
Tlayer_regularization_losses

Ustates
regularization_losses
trainable_variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xlayer_metrics
 	variables
Ymetrics
Zlayer_regularization_losses
!regularization_losses
"trainable_variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
2:0 2deepfake_detector/dense/kernel
,:* 2deepfake_detector/dense/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
­
[non_trainable_variables

\layers
]layer_metrics
&	variables
^metrics
_layer_regularization_losses
'regularization_losses
(trainable_variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
E:C  2)deepfake_detector/time_distributed/kernel
7:5  2'deepfake_detector/time_distributed/bias
=:;
 Ø@ 2'deepfake_detector/lstm/lstm_cell/kernel
E:C@ 21deepfake_detector/lstm/lstm_cell/recurrent_kernel
5:3@ 2%deepfake_detector/lstm/lstm_cell/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
®
`non_trainable_variables

alayers
blayer_metrics
4	variables
cmetrics
dlayer_regularization_losses
5regularization_losses
6trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
enon_trainable_variables

flayers
glayer_metrics
=	variables
hmetrics
ilayer_regularization_losses
>regularization_losses
?trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
°
jnon_trainable_variables

klayers
llayer_metrics
L	variables
mmetrics
nlayer_regularization_losses
Mregularization_losses
Ntrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ÍBÊ
"__inference__wrapped_model_7419524input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_deepfake_detector_layer_call_fn_7421834
3__inference_deepfake_detector_layer_call_fn_7421853
3__inference_deepfake_detector_layer_call_fn_7421872
3__inference_deepfake_detector_layer_call_fn_7421891³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
2__inference_time_distributed_layer_call_fn_7421978
2__inference_time_distributed_layer_call_fn_7421987
2__inference_time_distributed_layer_call_fn_7421996
2__inference_time_distributed_layer_call_fn_7422005À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422023
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422041
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_time_distributed_1_layer_call_fn_7422064
4__inference_time_distributed_1_layer_call_fn_7422069
4__inference_time_distributed_1_layer_call_fn_7422074
4__inference_time_distributed_1_layer_call_fn_7422079À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7422085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_reshape_for_rnn_layer_call_fn_7422090¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ç2ä
A__inference_lstm_layer_call_and_return_conditional_losses_7422241
A__inference_lstm_layer_call_and_return_conditional_losses_7422392
A__inference_lstm_layer_call_and_return_conditional_losses_7422543
A__inference_lstm_layer_call_and_return_conditional_losses_7422694Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
û2ø
&__inference_lstm_layer_call_fn_7422705
&__inference_lstm_layer_call_fn_7422716
&__inference_lstm_layer_call_fn_7422727
&__inference_lstm_layer_call_fn_7422738Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_7422744¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_flatten_layer_call_fn_7422749¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_7422760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_layer_call_fn_7422769¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
%__inference_signature_wrapper_7421087input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_layer_call_fn_7422789¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_average_pooling2d_layer_call_fn_7422804
3__inference_average_pooling2d_layer_call_fn_7422809¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_lstm_cell_layer_call_fn_7422890
+__inference_lstm_cell_layer_call_fn_7422907¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"__inference__wrapped_model_7419524s*+,-.$%<¢9
2¢/
-*
input_1ÿÿÿÿÿÿÿÿÿ<dd
ª "*ª'
%
output_1
output_1ñ
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿcc 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ11 
 É
3__inference_average_pooling2d_layer_call_fn_7422804R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3__inference_average_pooling2d_layer_call_fn_7422809[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿcc 
ª " ÿÿÿÿÿÿÿÿÿ11 ³
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780l*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿcc 
 
(__inference_conv2d_layer_call_fn_7422789_*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª " ÿÿÿÿÿÿÿÿÿcc º
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269h*+,-.$%?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p 
ª "¢

0
 º
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451h*+,-.$%?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p
ª "¢

0
 »
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633i*+,-.$%@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ<dd
p 
ª "¢

0
 »
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815i*+,-.$%@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ<dd
p
ª "¢

0
 
3__inference_deepfake_detector_layer_call_fn_7421834\*+,-.$%@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ<dd
p 
ª "
3__inference_deepfake_detector_layer_call_fn_7421853[*+,-.$%?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p 
ª "
3__inference_deepfake_detector_layer_call_fn_7421872[*+,-.$%?¢<
5¢2
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p
ª "
3__inference_deepfake_detector_layer_call_fn_7421891\*+,-.$%@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ<dd
p
ª "
B__inference_dense_layer_call_and_return_conditional_losses_7422760J$%&¢#
¢

inputs
ª "¢

0
 h
'__inference_dense_layer_call_fn_7422769=$%&¢#
¢

inputs
ª "
D__inference_flatten_layer_call_and_return_conditional_losses_7422744F&¢#
¢

inputs
ª "¢

0
 f
)__inference_flatten_layer_call_fn_74227499&¢#
¢

inputs
ª "Ê
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841ÿ,-.¢
x¢u
"
inputsÿÿÿÿÿÿÿÿÿ Ø
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 Ê
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873ÿ,-.¢
x¢u
"
inputsÿÿÿÿÿÿÿÿÿ Ø
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 
+__inference_lstm_cell_layer_call_fn_7422890ï,-.¢
x¢u
"
inputsÿÿÿÿÿÿÿÿÿ Ø
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
+__inference_lstm_cell_layer_call_fn_7422907ï,-.¢
x¢u
"
inputsÿÿÿÿÿÿÿÿÿ Ø
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÄ
A__inference_lstm_layer_call_and_return_conditional_losses_7422241,-.Q¢N
G¢D
63
1.
inputs/0 ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
A__inference_lstm_layer_call_and_return_conditional_losses_7422392,-.Q¢N
G¢D
63
1.
inputs/0 ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
A__inference_lstm_layer_call_and_return_conditional_losses_7422543],-.8¢5
.¢+

inputs< Ø

 
p 

 
ª "¢

0
 ¢
A__inference_lstm_layer_call_and_return_conditional_losses_7422694],-.8¢5
.¢+

inputs< Ø

 
p

 
ª "¢

0
 
&__inference_lstm_layer_call_fn_7422705r,-.Q¢N
G¢D
63
1.
inputs/0 ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_layer_call_fn_7422716r,-.Q¢N
G¢D
63
1.
inputs/0 ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ø

 
p

 
ª "ÿÿÿÿÿÿÿÿÿz
&__inference_lstm_layer_call_fn_7422727P,-.8¢5
.¢+

inputs< Ø

 
p 

 
ª "z
&__inference_lstm_layer_call_fn_7422738P,-.8¢5
.¢+

inputs< Ø

 
p

 
ª "±
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7422085a;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ<11 
ª ""¢

0< Ø
 
1__inference_reshape_for_rnn_layer_call_fn_7422090T;¢8
1¢.
,)
inputsÿÿÿÿÿÿÿÿÿ<11 
ª "< Ø§
%__inference_signature_wrapper_7421087~*+,-.$%G¢D
¢ 
=ª:
8
input_1-*
input_1ÿÿÿÿÿÿÿÿÿ<dd"*ª'
%
output_1
output_1Þ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422023L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 
 Þ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422041L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 
 Ë
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050xC¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<cc 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ<11 
 Ë
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059xC¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<cc 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ<11 
 µ
4__inference_time_distributed_1_layer_call_fn_7422064}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 µ
4__inference_time_distributed_1_layer_call_fn_7422069}L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ11 £
4__inference_time_distributed_1_layer_call_fn_7422074kC¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<cc 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿ<11 £
4__inference_time_distributed_1_layer_call_fn_7422079kC¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<cc 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿ<11 à
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915*+L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 à
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939*+L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc 
 Í
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954|*+C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ<cc 
 Í
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969|*+C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ<cc 
 ¸
2__inference_time_distributed_layer_call_fn_7421978*+L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc ¸
2__inference_time_distributed_layer_call_fn_7421987*+L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdd
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿcc ¥
2__inference_time_distributed_layer_call_fn_7421996o*+C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿ<cc ¥
2__inference_time_distributed_layer_call_fn_7422005o*+C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ<dd
p

 
ª "$!ÿÿÿÿÿÿÿÿÿ<cc 