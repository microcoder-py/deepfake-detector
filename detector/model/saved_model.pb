ќЋ
оД
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
╝
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
Џ
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
Ф
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
џ
TensorListReserve
element_shape"
shape_type
num_elements#
handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
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
ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8о▒
ў
deepfake_detector/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name deepfake_detector/dense/kernel
Љ
2deepfake_detector/dense/kernel/Read/ReadVariableOpReadVariableOpdeepfake_detector/dense/kernel*
_output_shapes

:*
dtype0
љ
deepfake_detector/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namedeepfake_detector/dense/bias
Ѕ
0deepfake_detector/dense/bias/Read/ReadVariableOpReadVariableOpdeepfake_detector/dense/bias*
_output_shapes
:*
dtype0
Х
)deepfake_detector/time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)deepfake_detector/time_distributed/kernel
»
=deepfake_detector/time_distributed/kernel/Read/ReadVariableOpReadVariableOp)deepfake_detector/time_distributed/kernel*&
_output_shapes
: *
dtype0
д
'deepfake_detector/time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'deepfake_detector/time_distributed/bias
Ъ
;deepfake_detector/time_distributed/bias/Read/ReadVariableOpReadVariableOp'deepfake_detector/time_distributed/bias*
_output_shapes
: *
dtype0
г
'deepfake_detector/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ап@*8
shared_name)'deepfake_detector/lstm/lstm_cell/kernel
Ц
;deepfake_detector/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp'deepfake_detector/lstm/lstm_cell/kernel* 
_output_shapes
:
ап@*
dtype0
Й
1deepfake_detector/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*B
shared_name31deepfake_detector/lstm/lstm_cell/recurrent_kernel
и
Edeepfake_detector/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp1deepfake_detector/lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
б
%deepfake_detector/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%deepfake_detector/lstm/lstm_cell/bias
Џ
9deepfake_detector/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOp%deepfake_detector/lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
Е 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*С
value┌BО Bл
ф
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
Г
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
Г
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
Г
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
Г
Fnon_trainable_variables

Glayers
Hlayer_metrics
	variables
Imetrics
Jlayer_regularization_losses
regularization_losses
trainable_variables
ј
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
╣
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
Г
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
Г
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
Г
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
Г
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
Г
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
њ
serving_default_input_1Placeholder*3
_output_shapes!
:         <dd*
dtype0*(
shape:         <dd
л
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
GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_7421087
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
К
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_7422951
Х
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
GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_7422982╝Ь
░
`
D__inference_flatten_layer_call_and_return_conditional_losses_7422744

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
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
ё
▓
&__inference_lstm_layer_call_fn_7422727

inputs
unknown:
ап@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЭ
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
GPU2*0J 8ѓ *J
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
:<ап: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
╠
ћ
'__inference_dense_layer_call_fn_7422769

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallВ
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
GPU2*0J 8ѓ *K
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
╩
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799

inputs
identityЏ
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:         11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         cc :W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
юX
є
A__inference_lstm_layer_call_and_return_conditional_losses_7420858

inputs<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhilec
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
B :У2
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
zeros_1/packed/1Ѕ
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
:<ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2ш
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpџ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpќ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1і
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpЌ
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
lstm_cell/split/split_dim├
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
lstm_cell/ReluЄ
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
lstm_cell/Relu_1І
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЯ
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
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shape▀
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Љ
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
transpose_1/permю
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

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:<ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
═	
г
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
Ѕ
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
═	
г
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
Ѕ
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
и
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Я
O
3__inference_average_pooling2d_layer_call_fn_7422804

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74196812
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
і
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
:<ап2	
Reshapea
IdentityIdentityReshape:output:0*
T0*$
_output_shapes
:<ап2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <11 :[ W
3
_output_shapes!
:         <11 
 
_user_specified_nameinputs
┌
╚
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
@: : : : :         :         : ::::: 
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Ж
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         cc 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Џ
Ю
(__inference_conv2d_layer_call_fn_7422789

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ХZ
ѕ
A__inference_lstm_layer_call_and_return_conditional_losses_7422241
inputs_0<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         2
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЄ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*6
_output_shapes$
":                   ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2■
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:         ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpБ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpЪ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/MatMul_1Њ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpа
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimу
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         2
lstm_cell/SigmoidЂ
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         2
lstm_cell/Sigmoid_1ѓ
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         2
lstm_cell/Reluљ
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         2
lstm_cell/mul_1Ё
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         2
lstm_cell/add_1Ђ
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         2
lstm_cell/Relu_1ћ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422157*
condR
while_cond_7422156*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
:         2

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:` \
6
_output_shapes$
":                   ап
"
_user_specified_name
inputs/0
і
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
:<ап2	
Reshapea
IdentityIdentityReshape:output:0*
T0*$
_output_shapes
:<ап2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <11 :[ W
3
_output_shapes!
:         <11 
 
_user_specified_nameinputs
Ц\
Ќ
)deepfake_detector_lstm_while_body_7419431J
Fdeepfake_detector_lstm_while_deepfake_detector_lstm_while_loop_counterP
Ldeepfake_detector_lstm_while_deepfake_detector_lstm_while_maximum_iterations,
(deepfake_detector_lstm_while_placeholder.
*deepfake_detector_lstm_while_placeholder_1.
*deepfake_detector_lstm_while_placeholder_2.
*deepfake_detector_lstm_while_placeholder_3I
Edeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1_0є
Ђdeepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0[
Gdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0:
ап@[
Ideepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:@V
Hdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:@)
%deepfake_detector_lstm_while_identity+
'deepfake_detector_lstm_while_identity_1+
'deepfake_detector_lstm_while_identity_2+
'deepfake_detector_lstm_while_identity_3+
'deepfake_detector_lstm_while_identity_4+
'deepfake_detector_lstm_while_identity_5G
Cdeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1Ѓ
deepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensorY
Edeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource:
ап@Y
Gdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource:@T
Fdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource:@ѕб=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOpб<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOpб>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOpы
Ndeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2P
Ndeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeО
@deepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЂdeepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0(deepfake_detector_lstm_while_placeholderWdeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype02B
@deepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItemє
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpGdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02>
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOpа
-deepfake_detector/lstm/while/lstm_cell/MatMulMatMulGdeepfake_detector/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Ddeepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2/
-deepfake_detector/lstm/while/lstm_cell/MatMulі
>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpIdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02@
>deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOpЅ
/deepfake_detector/lstm/while/lstm_cell/MatMul_1MatMul*deepfake_detector_lstm_while_placeholder_2Fdeepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@21
/deepfake_detector/lstm/while/lstm_cell/MatMul_1■
*deepfake_detector/lstm/while/lstm_cell/addAddV27deepfake_detector/lstm/while/lstm_cell/MatMul:product:09deepfake_detector/lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2,
*deepfake_detector/lstm/while/lstm_cell/addЃ
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpHdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02?
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOpІ
.deepfake_detector/lstm/while/lstm_cell/BiasAddBiasAdd.deepfake_detector/lstm/while/lstm_cell/add:z:0Edeepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@20
.deepfake_detector/lstm/while/lstm_cell/BiasAdd▓
6deepfake_detector/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6deepfake_detector/lstm/while/lstm_cell/split/split_dimи
,deepfake_detector/lstm/while/lstm_cell/splitSplit?deepfake_detector/lstm/while/lstm_cell/split/split_dim:output:07deepfake_detector/lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2.
,deepfake_detector/lstm/while/lstm_cell/split╦
.deepfake_detector/lstm/while/lstm_cell/SigmoidSigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:20
.deepfake_detector/lstm/while/lstm_cell/Sigmoid¤
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_1Sigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:22
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_1Ж
*deepfake_detector/lstm/while/lstm_cell/mulMul4deepfake_detector/lstm/while/lstm_cell/Sigmoid_1:y:0*deepfake_detector_lstm_while_placeholder_3*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/while/lstm_cell/mul┬
+deepfake_detector/lstm/while/lstm_cell/ReluRelu5deepfake_detector/lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2-
+deepfake_detector/lstm/while/lstm_cell/Reluч
,deepfake_detector/lstm/while/lstm_cell/mul_1Mul2deepfake_detector/lstm/while/lstm_cell/Sigmoid:y:09deepfake_detector/lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/mul_1­
,deepfake_detector/lstm/while/lstm_cell/add_1AddV2.deepfake_detector/lstm/while/lstm_cell/mul:z:00deepfake_detector/lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/add_1¤
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_2Sigmoid5deepfake_detector/lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:22
0deepfake_detector/lstm/while/lstm_cell/Sigmoid_2┴
-deepfake_detector/lstm/while/lstm_cell/Relu_1Relu0deepfake_detector/lstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2/
-deepfake_detector/lstm/while/lstm_cell/Relu_1 
,deepfake_detector/lstm/while/lstm_cell/mul_2Mul4deepfake_detector/lstm/while/lstm_cell/Sigmoid_2:y:0;deepfake_detector/lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2.
,deepfake_detector/lstm/while/lstm_cell/mul_2л
Adeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*deepfake_detector_lstm_while_placeholder_1(deepfake_detector_lstm_while_placeholder0deepfake_detector/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02C
Adeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItemі
"deepfake_detector/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"deepfake_detector/lstm/while/add/y┼
 deepfake_detector/lstm/while/addAddV2(deepfake_detector_lstm_while_placeholder+deepfake_detector/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2"
 deepfake_detector/lstm/while/addј
$deepfake_detector/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$deepfake_detector/lstm/while/add_1/yж
"deepfake_detector/lstm/while/add_1AddV2Fdeepfake_detector_lstm_while_deepfake_detector_lstm_while_loop_counter-deepfake_detector/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2$
"deepfake_detector/lstm/while/add_1К
%deepfake_detector/lstm/while/IdentityIdentity&deepfake_detector/lstm/while/add_1:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2'
%deepfake_detector/lstm/while/Identityы
'deepfake_detector/lstm/while/Identity_1IdentityLdeepfake_detector_lstm_while_deepfake_detector_lstm_while_maximum_iterations"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_1╔
'deepfake_detector/lstm/while/Identity_2Identity$deepfake_detector/lstm/while/add:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_2Ш
'deepfake_detector/lstm/while/Identity_3IdentityQdeepfake_detector/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes
: 2)
'deepfake_detector/lstm/while/Identity_3П
'deepfake_detector/lstm/while/Identity_4Identity0deepfake_detector/lstm/while/lstm_cell/mul_2:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/while/Identity_4П
'deepfake_detector/lstm/while/Identity_5Identity0deepfake_detector/lstm/while/lstm_cell/add_1:z:0"^deepfake_detector/lstm/while/NoOp*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/while/Identity_5╚
!deepfake_detector/lstm/while/NoOpNoOp>^deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp=^deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp?^deepfake_detector/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2#
!deepfake_detector/lstm/while/NoOp"ї
Cdeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1Edeepfake_detector_lstm_while_deepfake_detector_lstm_strided_slice_1_0"W
%deepfake_detector_lstm_while_identity.deepfake_detector/lstm/while/Identity:output:0"[
'deepfake_detector_lstm_while_identity_10deepfake_detector/lstm/while/Identity_1:output:0"[
'deepfake_detector_lstm_while_identity_20deepfake_detector/lstm/while/Identity_2:output:0"[
'deepfake_detector_lstm_while_identity_30deepfake_detector/lstm/while/Identity_3:output:0"[
'deepfake_detector_lstm_while_identity_40deepfake_detector/lstm/while/Identity_4:output:0"[
'deepfake_detector_lstm_while_identity_50deepfake_detector/lstm/while/Identity_5:output:0"њ
Fdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resourceHdeepfake_detector_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"ћ
Gdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resourceIdeepfake_detector_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"љ
Edeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resourceGdeepfake_detector_lstm_while_lstm_cell_matmul_readvariableop_resource_0"Ё
deepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensorЂdeepfake_detector_lstm_while_tensorarrayv2read_tensorlistgetitem_deepfake_detector_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2~
=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp=deepfake_detector/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2|
<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp<deepfake_detector/lstm/while/lstm_cell/MatMul/ReadVariableOp2ђ
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
Ё
P
4__inference_time_distributed_1_layer_call_fn_7422074

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74204652
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
р
M
1__inference_reshape_for_rnn_layer_call_fn_7422090

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:<ап* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:<ап2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <11 :[ W
3
_output_shapes!
:         <11 
 
_user_specified_nameinputs
┌
╚
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
@: : : : :         :         : ::::: 
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
─Ј
█
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815
input_1P
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
ап@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб%lstm/lstm_cell/BiasAdd/ReadVariableOpб$lstm/lstm_cell/MatMul/ReadVariableOpб&lstm/lstm_cell/MatMul_1/ReadVariableOpб
lstm/whileб.time_distributed/conv2d/BiasAdd/ReadVariableOpб-time_distributed/conv2d/Conv2D/ReadVariableOpЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeФ
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeП
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOpЄ
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2Dн
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpУ
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2!
time_distributed/conv2d/BiasAddе
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed/conv2d/ReluА
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2"
 time_distributed/Reshape_1/shapeп
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
time_distributed/Reshape_1Ю
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2"
 time_distributed/Reshape_2/shape▒
time_distributed/Reshape_2Reshapeinput_1)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/Reshape_2Ю
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape═
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshapeѓ
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPoolЦ
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   1   1       2$
"time_distributed_1/Reshape_1/shapeж
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
time_distributed_1/Reshape_1А
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2$
"time_distributed_1/Reshape_2/shapeМ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshape_2Њ
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape╗
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:<ап2
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
lstm/strided_slice/stackѓ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1ѓ
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2ђ
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
lstm/zeros/mul/yђ
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
B :У2
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
lstm/zeros/packed/1Ќ
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
lstm/zeros/Constђ

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
lstm/zeros_1/mul/yє
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
B :У2
lstm/zeros_1/Less/yЃ
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
lstm/zeros_1/packed/1Ю
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
lstm/zeros_1/Constѕ
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
lstm/transpose/permю
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:<ап2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1ѓ
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackє
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1є
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2ї
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1Ј
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm/TensorArrayV2/element_shapeк
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2╔
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeї
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorѓ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackє
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1є
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Њ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
lstm/strided_slice_2╝
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp«
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul└
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpф
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1ъ
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add╣
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpФ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAddѓ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimО
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/splitЃ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/SigmoidЄ
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1Ї
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/ReluЏ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1љ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1Є
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1Ъ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2Ў
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shape╠
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
	lstm/timeЅ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterФ

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

lstm/while┐
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeз
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackІ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm/strided_slice_3/stackє
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1є
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2»
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3Ѓ
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm░
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
valueB"       2
flatten/ConstЇ
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpј
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpљ
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

Identityы
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2<
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
:         <dd
!
_user_specified_name	input_1
№	
┤
3__inference_deepfake_detector_layer_call_fn_7421891
input_1!
unknown: 
	unknown_0: 
	unknown_1:
ап@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identityѕбStatefulPartitionedCall║
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
GPU2*0J 8ѓ *W
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
-:         <dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         <dd
!
_user_specified_name	input_1
═	
г
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
Ѕ
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
№
Ђ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7420019

inputs

states
states_12
matmul_readvariableop_resource:
ап@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_2Ў
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
A:         ап:         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
╦
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7420449

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2
Reshape_1/shapeћ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
╦B
н	
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
ап@I
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
ап@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@ѕб+lstm/while/lstm_cell/BiasAdd/ReadVariableOpб*lstm/while/lstm_cell/MatMul/ReadVariableOpб,lstm/while/lstm_cell/MatMul_1/ReadVariableOp═
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemл
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpп
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulн
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp┴
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1Х
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/add═
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp├
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAddј
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim№
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/splitЋ
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/SigmoidЎ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1б
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mulї
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu│
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1е
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1Ў
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2І
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1и
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2Ш
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
lstm/while/add_1/yЈ
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЌ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Ђ
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2«
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ћ
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4Ћ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5Ь
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
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"╝
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
э
Ѓ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841

inputs
states_0
states_12
matmul_readvariableop_resource:
ап@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_2Ў
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
A:         ап:         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
Ж
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_7419549

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         cc 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
╦
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2
Reshape_1/shapeћ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
Х
╚
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
 E
щ
A__inference_lstm_layer_call_and_return_conditional_losses_7420166

inputs%
lstm_cell_7420084:
ап@#
lstm_cell_7420086:@
lstm_cell_7420088:@
identityѕб!lstm_cell/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         2
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputstranspose/perm:output:0*
T0*6
_output_shapes$
":                   ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2■
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:         ап*
shrink_axis_mask2
strided_slice_2њ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7420084lstm_cell_7420086lstm_cell_7420088*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192#
!lstm_cell/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7420084lstm_cell_7420086lstm_cell_7420088*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7420097*
condR
while_cond_7420096*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
:         2

Identityz
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:^ Z
6
_output_shapes$
":                   ап
 
_user_specified_nameinputs
╩
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7419712

inputs
identityЏ
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:         11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         cc :W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
└
┤
&__inference_lstm_layer_call_fn_7422705
inputs_0
unknown:
ап@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74199562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
":                   ап
"
_user_specified_name
inputs/0
њ=
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeН
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:         ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp═
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpХ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/MatMul_1Ф
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpИ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim 
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
while/lstm_cell/splitЈ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         2
while/lstm_cell/SigmoidЊ
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         2
while/lstm_cell/Sigmoid_1Ќ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         2
while/lstm_cell/mulє
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         2
while/lstm_cell/Reluе
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         2
while/lstm_cell/add_1Њ
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         2
while/lstm_cell/Sigmoid_2Ё
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         2
while/lstm_cell/Relu_1г
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3і
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_4і
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
Џ
E
)__inference_flatten_layer_call_fn_7422749

inputs
identity╝
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
GPU2*0J 8ѓ *M
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
И;
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╠
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp─
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpГ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1б
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp»
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim█
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/splitє
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoidі
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1ј
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/ReluЪ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1ћ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1і
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1Б
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ђ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4Ђ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
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
И
з
+__inference_lstm_cell_layer_call_fn_7422890

inputs
states_0
states_1
unknown:
ап@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         2

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
A:         ап:         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
╦

з
B__inference_dense_layer_call_and_return_conditional_losses_7422760

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMulї
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
И
з
+__inference_lstm_cell_layer_call_fn_7422907

inputs
states_0
states_1
unknown:
ап@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         2

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
A:         ап:         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
Ж
█
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

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╣
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╦
value┴BЙB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesў
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slicesЊ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_deepfake_detector_dense_kernel_read_readvariableop7savev2_deepfake_detector_dense_bias_read_readvariableopDsavev2_deepfake_detector_time_distributed_kernel_read_readvariableopBsavev2_deepfake_detector_time_distributed_bias_read_readvariableopBsavev2_deepfake_detector_lstm_lstm_cell_kernel_read_readvariableopLsavev2_deepfake_detector_lstm_lstm_cell_recurrent_kernel_read_readvariableop@savev2_deepfake_detector_lstm_lstm_cell_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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
ап@:@:@: 2(
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
ап@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: 
ў
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpD
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
strided_slice/stack_2Р
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
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeЮ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  cc 2
	Reshape_1ѓ
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
х	
д
%__inference_signature_wrapper_7421087
input_1!
unknown: 
	unknown_0: 
	unknown_1:
ап@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identityѕбStatefulPartitionedCallј
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
GPU2*0J 8ѓ *+
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
-:         <dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         <dd
!
_user_specified_name	input_1
ф
P
4__inference_time_distributed_1_layer_call_fn_7422069

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&                  11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74197542
PartitionedCallЂ
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs
Ь
Н
M__inference_time_distributed_layer_call_and_return_conditional_losses_7419612

inputs(
conv2d_7419600: 
conv2d_7419602: 
identityѕбconv2d/StatefulPartitionedCallD
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
strided_slice/stack_2Р
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
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeб
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_7419600conv2d_7419602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeФ
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  cc 2
	Reshape_1ѓ
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

Identityo
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
В	
│
3__inference_deepfake_detector_layer_call_fn_7421853

inputs!
unknown: 
	unknown_0: 
	unknown_1:
ап@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identityѕбStatefulPartitionedCall╣
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
GPU2*0J 8ѓ *W
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
-:         <dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
╦B
н	
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
ап@I
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
ап@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@ѕб+lstm/while/lstm_cell/BiasAdd/ReadVariableOpб*lstm/while/lstm_cell/MatMul/ReadVariableOpб,lstm/while/lstm_cell/MatMul_1/ReadVariableOp═
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemл
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpп
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulн
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp┴
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1Х
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/add═
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp├
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAddј
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim№
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/splitЋ
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/SigmoidЎ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1б
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mulї
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu│
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1е
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1Ў
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2І
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1и
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2Ш
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
lstm/while/add_1/yЈ
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЌ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Ђ
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2«
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ћ
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4Ћ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5Ь
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
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"╝
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
┌
╚
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
@: : : : :         :         : ::::: 
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
э
Ѓ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873

inputs
states_0
states_12
matmul_readvariableop_resource:
ап@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_2Ў
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
A:         ап:         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ХZ
ѕ
A__inference_lstm_layer_call_and_return_conditional_losses_7422392
inputs_0<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         2
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЄ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*6
_output_shapes$
":                   ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2■
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:         ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpБ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpЪ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/MatMul_1Њ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpа
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimу
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         2
lstm_cell/SigmoidЂ
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         2
lstm_cell/Sigmoid_1ѓ
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         2
lstm_cell/Reluљ
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         2
lstm_cell/mul_1Ё
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         2
lstm_cell/add_1Ђ
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         2
lstm_cell/Relu_1ћ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7422308*
condR
while_cond_7422307*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
:         2

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:` \
6
_output_shapes$
":                   ап
"
_user_specified_name
inputs/0
╦
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7420918

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2
Reshape_1/shapeћ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
џ	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
value B"    <   1   1       2
Reshape_1/shapeЮ
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
╦

з
B__inference_dense_layer_call_and_return_conditional_losses_7420654

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMulї
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
с
Д
2__inference_time_distributed_layer_call_fn_7421987

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&                  cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74196122
StatefulPartitionedCallљ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
Ё
P
4__inference_time_distributed_1_layer_call_fn_7422079

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74208892
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
Х
╚
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
└Ј
┌
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
ап@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб%lstm/lstm_cell/BiasAdd/ReadVariableOpб$lstm/lstm_cell/MatMul/ReadVariableOpб&lstm/lstm_cell/MatMul_1/ReadVariableOpб
lstm/whileб.time_distributed/conv2d/BiasAdd/ReadVariableOpб-time_distributed/conv2d/Conv2D/ReadVariableOpЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeф
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeП
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOpЄ
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2Dн
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpУ
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2!
time_distributed/conv2d/BiasAddе
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed/conv2d/ReluА
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2"
 time_distributed/Reshape_1/shapeп
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
time_distributed/Reshape_1Ю
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2"
 time_distributed/Reshape_2/shape░
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/Reshape_2Ю
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape═
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshapeѓ
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPoolЦ
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   1   1       2$
"time_distributed_1/Reshape_1/shapeж
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
time_distributed_1/Reshape_1А
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2$
"time_distributed_1/Reshape_2/shapeМ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshape_2Њ
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape╗
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:<ап2
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
lstm/strided_slice/stackѓ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1ѓ
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2ђ
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
lstm/zeros/mul/yђ
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
B :У2
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
lstm/zeros/packed/1Ќ
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
lstm/zeros/Constђ

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
lstm/zeros_1/mul/yє
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
B :У2
lstm/zeros_1/Less/yЃ
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
lstm/zeros_1/packed/1Ю
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
lstm/zeros_1/Constѕ
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
lstm/transpose/permю
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:<ап2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1ѓ
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackє
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1є
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2ї
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1Ј
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm/TensorArrayV2/element_shapeк
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2╔
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeї
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorѓ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackє
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1є
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Њ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
lstm/strided_slice_2╝
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp«
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul└
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpф
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1ъ
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add╣
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpФ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAddѓ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimО
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/splitЃ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/SigmoidЄ
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1Ї
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/ReluЏ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1љ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1Є
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1Ъ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2Ў
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shape╠
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
	lstm/timeЅ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterФ

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

lstm/while┐
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeз
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackІ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm/strided_slice_3/stackє
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1є
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2»
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3Ѓ
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm░
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
valueB"       2
flatten/ConstЇ
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpј
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpљ
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

Identityы
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2<
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
:         <dd
 
_user_specified_nameinputs
Х
╚
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
А%
М
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
ап@+
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
ап@)
while_lstm_cell_7419913:@%
while_lstm_cell_7419915:@ѕб'while/lstm_cell/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeН
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:         ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemо
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7419911_0while_lstm_cell_7419913_0while_lstm_cell_7419915_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732)
'while/lstm_cell/StatefulPartitionedCallЗ
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3А
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_4А
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_5ё

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
юX
є
A__inference_lstm_layer_call_and_return_conditional_losses_7422543

inputs<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhilec
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
B :У2
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
zeros_1/packed/1Ѕ
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
:<ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2ш
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpџ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpќ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1і
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpЌ
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
lstm_cell/split/split_dim├
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
lstm_cell/ReluЄ
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
lstm_cell/Relu_1І
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЯ
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
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shape▀
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Љ
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
transpose_1/permю
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

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:<ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
№	
┤
3__inference_deepfake_detector_layer_call_fn_7421834
input_1!
unknown: 
	unknown_0: 
	unknown_1:
ап@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identityѕбStatefulPartitionedCall║
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
GPU2*0J 8ѓ *W
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
-:         <dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         <dd
!
_user_specified_name	input_1
═	
г
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
Ѕ
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
└Ј
┌
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
ап@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб%lstm/lstm_cell/BiasAdd/ReadVariableOpб$lstm/lstm_cell/MatMul/ReadVariableOpб&lstm/lstm_cell/MatMul_1/ReadVariableOpб
lstm/whileб.time_distributed/conv2d/BiasAdd/ReadVariableOpб-time_distributed/conv2d/Conv2D/ReadVariableOpЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeф
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeП
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOpЄ
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2Dн
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpУ
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2!
time_distributed/conv2d/BiasAddе
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed/conv2d/ReluА
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2"
 time_distributed/Reshape_1/shapeп
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
time_distributed/Reshape_1Ю
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2"
 time_distributed/Reshape_2/shape░
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/Reshape_2Ю
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape═
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshapeѓ
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPoolЦ
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   1   1       2$
"time_distributed_1/Reshape_1/shapeж
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
time_distributed_1/Reshape_1А
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2$
"time_distributed_1/Reshape_2/shapeМ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshape_2Њ
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape╗
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:<ап2
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
lstm/strided_slice/stackѓ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1ѓ
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2ђ
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
lstm/zeros/mul/yђ
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
B :У2
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
lstm/zeros/packed/1Ќ
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
lstm/zeros/Constђ

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
lstm/zeros_1/mul/yє
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
B :У2
lstm/zeros_1/Less/yЃ
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
lstm/zeros_1/packed/1Ю
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
lstm/zeros_1/Constѕ
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
lstm/transpose/permю
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:<ап2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1ѓ
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackє
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1є
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2ї
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1Ј
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm/TensorArrayV2/element_shapeк
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2╔
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeї
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorѓ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackє
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1є
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Њ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
lstm/strided_slice_2╝
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp«
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul└
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpф
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1ъ
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add╣
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpФ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAddѓ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimО
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/splitЃ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/SigmoidЄ
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1Ї
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/ReluЏ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1љ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1Є
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1Ъ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2Ў
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shape╠
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
	lstm/timeЅ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterФ

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

lstm/while┐
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeз
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackІ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm/strided_slice_3/stackє
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1є
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2»
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3Ѓ
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm░
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
valueB"       2
flatten/ConstЇ
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpј
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpљ
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

Identityы
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2<
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
:         <dd
 
_user_specified_nameinputs
─Ј
█
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633
input_1P
6time_distributed_conv2d_conv2d_readvariableop_resource: E
7time_distributed_conv2d_biasadd_readvariableop_resource: A
-lstm_lstm_cell_matmul_readvariableop_resource:
ап@A
/lstm_lstm_cell_matmul_1_readvariableop_resource:@<
.lstm_lstm_cell_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб%lstm/lstm_cell/BiasAdd/ReadVariableOpб$lstm/lstm_cell/MatMul/ReadVariableOpб&lstm/lstm_cell/MatMul_1/ReadVariableOpб
lstm/whileб.time_distributed/conv2d/BiasAdd/ReadVariableOpб-time_distributed/conv2d/Conv2D/ReadVariableOpЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeФ
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeП
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOpЄ
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2Dн
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpУ
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2!
time_distributed/conv2d/BiasAddе
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed/conv2d/ReluА
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2"
 time_distributed/Reshape_1/shapeп
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
time_distributed/Reshape_1Ю
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2"
 time_distributed/Reshape_2/shape▒
time_distributed/Reshape_2Reshapeinput_1)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/Reshape_2Ю
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape═
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshapeѓ
,time_distributed_1/average_pooling2d/AvgPoolAvgPool#time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2.
,time_distributed_1/average_pooling2d/AvgPoolЦ
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   1   1       2$
"time_distributed_1/Reshape_1/shapeж
time_distributed_1/Reshape_1Reshape5time_distributed_1/average_pooling2d/AvgPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
time_distributed_1/Reshape_1А
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2$
"time_distributed_1/Reshape_2/shapeМ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/Reshape_2Њ
reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
reshape_for_rnn/Reshape/shape╗
reshape_for_rnn/ReshapeReshape%time_distributed_1/Reshape_1:output:0&reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:<ап2
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
lstm/strided_slice/stackѓ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1ѓ
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2ђ
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
lstm/zeros/mul/yђ
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
B :У2
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
lstm/zeros/packed/1Ќ
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
lstm/zeros/Constђ

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
lstm/zeros_1/mul/yє
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
B :У2
lstm/zeros_1/Less/yЃ
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
lstm/zeros_1/packed/1Ю
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
lstm/zeros_1/Constѕ
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
lstm/transpose/permю
lstm/transpose	Transpose reshape_for_rnn/Reshape:output:0lstm/transpose/perm:output:0*
T0*$
_output_shapes
:<ап2
lstm/transposeq
lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2
lstm/Shape_1ѓ
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stackє
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1є
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2ї
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1Ј
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm/TensorArrayV2/element_shapeк
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2╔
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeї
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensorѓ
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stackє
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1є
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2Њ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
lstm/strided_slice_2╝
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp«
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul└
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpф
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/MatMul_1ъ
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/add╣
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpФ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/lstm_cell/BiasAddѓ
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dimО
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/lstm_cell/splitЃ
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/SigmoidЄ
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_1Ї
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mulz
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/lstm_cell/ReluЏ
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_1љ
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/add_1Є
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2
lstm/lstm_cell/Sigmoid_2y
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/lstm_cell/Relu_1Ъ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/lstm_cell/mul_2Ў
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm/TensorArrayV2_1/element_shape╠
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
	lstm/timeЅ
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterФ

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

lstm/while┐
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeз
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStackІ
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm/strided_slice_3/stackє
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1є
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2»
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
lstm/strided_slice_3Ѓ
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm░
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
valueB"       2
flatten/ConstЇ
flatten/ReshapeReshapelstm/strided_slice_3:output:0flatten/Const:output:0*
T0*
_output_shapes

:2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpј
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpљ
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

Identityы
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2<
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
:         <dd
!
_user_specified_name	input_1
┐
Д
2__inference_time_distributed_layer_call_fn_7421996

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74204492
StatefulPartitionedCallЄ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
Ь
Н
M__inference_time_distributed_layer_call_and_return_conditional_losses_7419562

inputs(
conv2d_7419550: 
conv2d_7419552: 
identityѕбconv2d/StatefulPartitionedCallD
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
strided_slice/stack_2Р
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
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeб
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_7419550conv2d_7419552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_74195492 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeФ
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  cc 2
	Reshape_1ѓ
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

Identityo
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
В	
│
3__inference_deepfake_detector_layer_call_fn_7421872

inputs!
unknown: 
	unknown_0: 
	unknown_1:
ап@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:
identityѕбStatefulPartitionedCall╣
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
GPU2*0J 8ѓ *W
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
-:         <dd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
и
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7419681

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┌
╚
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
@: : : : :         :         : ::::: 
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
А%
М
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
ап@+
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
ап@)
while_lstm_cell_7420123:@%
while_lstm_cell_7420125:@ѕб'while/lstm_cell/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeН
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:         ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemо
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7420121_0while_lstm_cell_7420123_0while_lstm_cell_7420125_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74200192)
'while/lstm_cell/StatefulPartitionedCallЗ
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3А
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_4А
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_5ё

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ј%
Ь
#__inference__traced_restore_7422982
file_prefixA
/assignvariableop_deepfake_detector_dense_kernel:=
/assignvariableop_1_deepfake_detector_dense_bias:V
<assignvariableop_2_deepfake_detector_time_distributed_kernel: H
:assignvariableop_3_deepfake_detector_time_distributed_bias: N
:assignvariableop_4_deepfake_detector_lstm_lstm_cell_kernel:
ап@V
Dassignvariableop_5_deepfake_detector_lstm_lstm_cell_recurrent_kernel:@F
8assignvariableop_6_deepfake_detector_lstm_lstm_cell_bias:@

identity_8ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6┐
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╦
value┴BЙB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slicesМ
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

Identity«
AssignVariableOpAssignVariableOp/assignvariableop_deepfake_detector_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1┤
AssignVariableOp_1AssignVariableOp/assignvariableop_1_deepfake_detector_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┴
AssignVariableOp_2AssignVariableOp<assignvariableop_2_deepfake_detector_time_distributed_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3┐
AssignVariableOp_3AssignVariableOp:assignvariableop_3_deepfake_detector_time_distributed_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4┐
AssignVariableOp_4AssignVariableOp:assignvariableop_4_deepfake_detector_lstm_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╔
AssignVariableOp_5AssignVariableOpDassignvariableop_5_deepfake_detector_lstm_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6й
AssignVariableOp_6AssignVariableOp8assignvariableop_6_deepfake_detector_lstm_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpщ

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7c

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_8с
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
┐
Д
2__inference_time_distributed_layer_call_fn_7422005

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74209182
StatefulPartitionedCallЄ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
Х
╚
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
џ	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
value B"    <   1   1       2
Reshape_1/shapeЮ
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
╦B
н	
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
ап@I
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
ап@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@ѕб+lstm/while/lstm_cell/BiasAdd/ReadVariableOpб*lstm/while/lstm_cell/MatMul/ReadVariableOpб,lstm/while/lstm_cell/MatMul_1/ReadVariableOp═
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemл
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpп
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulн
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp┴
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1Х
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/add═
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp├
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAddј
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim№
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/splitЋ
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/SigmoidЎ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1б
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mulї
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu│
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1е
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1Ў
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2І
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1и
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2Ш
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
lstm/while/add_1/yЈ
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЌ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Ђ
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2«
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ћ
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4Ћ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5Ь
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
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"╝
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
юX
є
A__inference_lstm_layer_call_and_return_conditional_losses_7420627

inputs<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhilec
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
B :У2
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
zeros_1/packed/1Ѕ
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
:<ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2ш
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpџ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpќ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1і
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpЌ
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
lstm_cell/split/split_dim├
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
lstm_cell/ReluЄ
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
lstm_cell/Relu_1І
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЯ
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
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shape▀
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Љ
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
transpose_1/permю
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

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:<ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
ф
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
strided_slice/stack_2Р
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
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
ReshapeЁ
!average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122#
!average_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape«
	Reshape_1Reshape*average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs
▄"
¤
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7420974

inputs2
time_distributed_7420949: &
time_distributed_7420951:  
lstm_7420960:
ап@
lstm_7420962:@
lstm_7420964:@
dense_7420968:
dense_7420970:
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallб(time_distributed/StatefulPartitionedCall╬
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_7420949time_distributed_7420951*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74209182*
(time_distributed/StatefulPartitionedCallЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeф
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeГ
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74208892$
"time_distributed_1/PartitionedCallЮ
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape█
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/ReshapeЈ
reshape_for_rnn/PartitionedCallPartitionedCall+time_distributed_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:<ап* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752!
reshape_for_rnn/PartitionedCall»
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
GPU2*0J 8ѓ *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74208582
lstm/StatefulPartitionedCallв
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
GPU2*0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_74206412
flatten/PartitionedCallю
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
GPU2*0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_74206542
dense/StatefulPartitionedCallx
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

IdentityИ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
с
Д
2__inference_time_distributed_layer_call_fn_7421978

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&                  cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74195622
StatefulPartitionedCallљ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
юX
є
A__inference_lstm_layer_call_and_return_conditional_losses_7422694

inputs<
(lstm_cell_matmul_readvariableop_resource:
ап@<
*lstm_cell_matmul_1_readvariableop_resource:@7
)lstm_cell_biasadd_readvariableop_resource:@
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhilec
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
B :У2
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
zeros_1/packed/1Ѕ
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
:<ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2ш
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2
strided_slice_2Г
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02!
lstm_cell/MatMul/ReadVariableOpџ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul▒
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOpќ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm_cell/MatMul_1і
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm_cell/addф
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOpЌ
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
lstm_cell/split/split_dim├
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
lstm_cell/ReluЄ
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
lstm_cell/Relu_1І
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm_cell/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЯ
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
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shape▀
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Љ
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
transpose_1/permю
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

Identity┐
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:<ап: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
И;
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╠
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp─
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpГ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1б
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp»
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim█
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/splitє
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoidі
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1ј
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/ReluЪ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1ћ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1і
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1Б
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ђ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4Ђ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
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
 E
щ
A__inference_lstm_layer_call_and_return_conditional_losses_7419956

inputs%
lstm_cell_7419874:
ап@#
lstm_cell_7419876:@
lstm_cell_7419878:@
identityѕб!lstm_cell/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         2
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputstranspose/perm:output:0*
T0*6
_output_shapes$
":                   ап2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2■
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*)
_output_shapes
:         ап*
shrink_axis_mask2
strided_slice_2њ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7419874lstm_cell_7419876lstm_cell_7419878*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_74198732#
!lstm_cell/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
TensorArrayV2_1/element_shapeИ
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7419874lstm_cell_7419876lstm_cell_7419878*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_7419887*
condR
while_cond_7419886*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
:         2

Identityz
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:^ Z
6
_output_shapes$
":                   ап
 
_user_specified_nameinputs
И;
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╠
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp─
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpГ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1б
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp»
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim█
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/splitє
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoidі
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1ј
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/ReluЪ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1ћ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1і
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1Б
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ђ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4Ђ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
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
з
O
3__inference_average_pooling2d_layer_call_fn_7422809

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         cc :W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
╦
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       2
Reshape_1/shapeћ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2
	Reshape_1y
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:         <cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         <dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
▄"
¤
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7420661

inputs2
time_distributed_7420450: &
time_distributed_7420452:  
lstm_7420628:
ап@
lstm_7420630:@
lstm_7420632:@
dense_7420655:
dense_7420657:
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallб(time_distributed/StatefulPartitionedCall╬
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_7420450time_distributed_7420452*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <cc *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_time_distributed_layer_call_and_return_conditional_losses_74204492*
(time_distributed/StatefulPartitionedCallЎ
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      2 
time_distributed/Reshape/shapeф
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2
time_distributed/ReshapeГ
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         <11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74204652$
"time_distributed_1/PartitionedCallЮ
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2"
 time_distributed_1/Reshape/shape█
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2
time_distributed_1/ReshapeЈ
reshape_for_rnn/PartitionedCallPartitionedCall+time_distributed_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:<ап* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_74204752!
reshape_for_rnn/PartitionedCall»
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
GPU2*0J 8ѓ *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74206272
lstm/StatefulPartitionedCallв
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
GPU2*0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_74206412
flatten/PartitionedCallю
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
GPU2*0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_74206542
dense/StatefulPartitionedCallx
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

IdentityИ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:[ W
3
_output_shapes!
:         <dd
 
_user_specified_nameinputs
ё
▓
&__inference_lstm_layer_call_fn_7422738

inputs
unknown:
ап@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЭ
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
GPU2*0J 8ѓ *J
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
:<ап: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:<ап
 
_user_specified_nameinputs
╦B
н	
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
ап@I
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
ап@G
5lstm_while_lstm_cell_matmul_1_readvariableop_resource:@B
4lstm_while_lstm_cell_biasadd_readvariableop_resource:@ѕб+lstm/while/lstm_cell/BiasAdd/ReadVariableOpб*lstm/while/lstm_cell/MatMul/ReadVariableOpб,lstm/while/lstm_cell/MatMul_1/ReadVariableOp═
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeЖ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemл
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpп
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMulн
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp┴
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/MatMul_1Х
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/add═
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp├
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
lstm/while/lstm_cell/BiasAddј
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim№
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm/while/lstm_cell/splitЋ
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/SigmoidЎ
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_1б
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mulї
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu│
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_1е
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/add_1Ў
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*
_output_shapes

:2 
lstm/while/lstm_cell/Sigmoid_2І
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/Relu_1и
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
lstm/while/lstm_cell/mul_2Ш
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
lstm/while/add_1/yЈ
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/IdentityЌ
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1Ђ
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2«
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3Ћ
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_4Ћ
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes

:2
lstm/while/Identity_5Ь
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
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"╝
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
ў
б
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpD
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
strided_slice/stack_2Р
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
valueB"    d   d      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         dd2	
Reshapeф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp├
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeЮ
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  cc 2
	Reshape_1ѓ
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&                  cc 2

IdentityЇ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&                  dd: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&                  dd
 
_user_specified_nameinputs
њ=
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeН
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*)
_output_shapes
:         ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp═
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpХ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/MatMul_1Ф
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpИ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim 
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
while/lstm_cell/splitЈ
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         2
while/lstm_cell/SigmoidЊ
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         2
while/lstm_cell/Sigmoid_1Ќ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         2
while/lstm_cell/mulє
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         2
while/lstm_cell/Reluе
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         2
while/lstm_cell/mul_1Ю
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         2
while/lstm_cell/add_1Њ
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         2
while/lstm_cell/Sigmoid_2Ё
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         2
while/lstm_cell/Relu_1г
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3і
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_4і
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
џ	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7420889

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
value B"    <   1   1       2
Reshape_1/shapeЮ
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
№
Ђ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7419873

inputs

states
states_12
matmul_readvariableop_resource:
ап@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity_2Ў
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
A:         ап:         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
)
_output_shapes
:         ап
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
џ	
k
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7420465

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
value B"    <   1   1       2
Reshape_1/shapeЮ
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:         <11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <cc :[ W
3
_output_shapes!
:         <cc 
 
_user_specified_nameinputs
Ж
ћ
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
с
!deepfake_detector/lstm/while/LessLess(deepfake_detector_lstm_while_placeholderHdeepfake_detector_lstm_while_less_deepfake_detector_lstm_strided_slice_1*
T0*
_output_shapes
: 2#
!deepfake_detector/lstm/while/Lessб
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
└
┤
&__inference_lstm_layer_call_fn_7422716
inputs_0
unknown:
ап@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_74201662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   ап: : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
6
_output_shapes$
":                   ап
"
_user_specified_name
inputs/0
Т
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
strided_slice/stack_2Р
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
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeд
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs
░
`
D__inference_flatten_layer_call_and_return_conditional_losses_7420641

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
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
ф
P
4__inference_time_distributed_1_layer_call_fn_7422064

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&                  11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_74197212
PartitionedCallЂ
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs
И;
┤
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
ап@D
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
ап@B
0while_lstm_cell_matmul_1_readvariableop_resource:@=
/while_lstm_cell_biasadd_readvariableop_resource:@ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╠
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0* 
_output_shapes
:
ап*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem┴
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
ап@*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp─
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul┼
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpГ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/MatMul_1б
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2
while/lstm_cell/addЙ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp»
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2
while/lstm_cell/BiasAddё
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim█
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2
while/lstm_cell/splitє
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoidі
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_1ј
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*
_output_shapes

:2
while/lstm_cell/mul}
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*
_output_shapes

:2
while/lstm_cell/ReluЪ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_1ћ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/add_1і
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*
_output_shapes

:2
while/lstm_cell/Sigmoid_2|
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2
while/lstm_cell/Relu_1Б
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2
while/lstm_cell/mul_2П
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ђ
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_4Ђ
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*
_output_shapes

:2
while/Identity_5Н

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
while_strided_slice_1while_strided_slice_1_0"е
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
П┐
й
"__inference__wrapped_model_7419524
input_1b
Hdeepfake_detector_time_distributed_conv2d_conv2d_readvariableop_resource: W
Ideepfake_detector_time_distributed_conv2d_biasadd_readvariableop_resource: S
?deepfake_detector_lstm_lstm_cell_matmul_readvariableop_resource:
ап@S
Adeepfake_detector_lstm_lstm_cell_matmul_1_readvariableop_resource:@N
@deepfake_detector_lstm_lstm_cell_biasadd_readvariableop_resource:@H
6deepfake_detector_dense_matmul_readvariableop_resource:E
7deepfake_detector_dense_biasadd_readvariableop_resource:
identityѕб.deepfake_detector/dense/BiasAdd/ReadVariableOpб-deepfake_detector/dense/MatMul/ReadVariableOpб7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOpб6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOpб8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOpбdeepfake_detector/lstm/whileб@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOpб?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOpй
0deepfake_detector/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      22
0deepfake_detector/time_distributed/Reshape/shapeр
*deepfake_detector/time_distributed/ReshapeReshapeinput_19deepfake_detector/time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:         dd2,
*deepfake_detector/time_distributed/ReshapeЊ
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOpHdeepfake_detector_time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02A
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp¤
0deepfake_detector/time_distributed/conv2d/Conv2DConv2D3deepfake_detector/time_distributed/Reshape:output:0Gdeepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc *
paddingVALID*
strides
22
0deepfake_detector/time_distributed/conv2d/Conv2Dі
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOpIdeepfake_detector_time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp░
1deepfake_detector/time_distributed/conv2d/BiasAddBiasAdd9deepfake_detector/time_distributed/conv2d/Conv2D:output:0Hdeepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         cc 23
1deepfake_detector/time_distributed/conv2d/BiasAddя
.deepfake_detector/time_distributed/conv2d/ReluRelu:deepfake_detector/time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         cc 20
.deepfake_detector/time_distributed/conv2d/Relu┼
2deepfake_detector/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   c   c       24
2deepfake_detector/time_distributed/Reshape_1/shapeа
,deepfake_detector/time_distributed/Reshape_1Reshape<deepfake_detector/time_distributed/conv2d/Relu:activations:0;deepfake_detector/time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <cc 2.
,deepfake_detector/time_distributed/Reshape_1┴
2deepfake_detector/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    d   d      24
2deepfake_detector/time_distributed/Reshape_2/shapeу
,deepfake_detector/time_distributed/Reshape_2Reshapeinput_1;deepfake_detector/time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         dd2.
,deepfake_detector/time_distributed/Reshape_2┴
2deepfake_detector/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       24
2deepfake_detector/time_distributed_1/Reshape/shapeЋ
,deepfake_detector/time_distributed_1/ReshapeReshape5deepfake_detector/time_distributed/Reshape_1:output:0;deepfake_detector/time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         cc 2.
,deepfake_detector/time_distributed_1/ReshapeИ
>deepfake_detector/time_distributed_1/average_pooling2d/AvgPoolAvgPool5deepfake_detector/time_distributed_1/Reshape:output:0*
T0*/
_output_shapes
:         11 *
ksize
*
paddingVALID*
strides
2@
>deepfake_detector/time_distributed_1/average_pooling2d/AvgPool╔
4deepfake_detector/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"    <   1   1       26
4deepfake_detector/time_distributed_1/Reshape_1/shape▒
.deepfake_detector/time_distributed_1/Reshape_1ReshapeGdeepfake_detector/time_distributed_1/average_pooling2d/AvgPool:output:0=deepfake_detector/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:         <11 20
.deepfake_detector/time_distributed_1/Reshape_1┼
4deepfake_detector/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    c   c       26
4deepfake_detector/time_distributed_1/Reshape_2/shapeЏ
.deepfake_detector/time_distributed_1/Reshape_2Reshape5deepfake_detector/time_distributed/Reshape_1:output:0=deepfake_detector/time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:         cc 20
.deepfake_detector/time_distributed_1/Reshape_2и
/deepfake_detector/reshape_for_rnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 21
/deepfake_detector/reshape_for_rnn/Reshape/shapeЃ
)deepfake_detector/reshape_for_rnn/ReshapeReshape7deepfake_detector/time_distributed_1/Reshape_1:output:08deepfake_detector/reshape_for_rnn/Reshape/shape:output:0*
T0*$
_output_shapes
:<ап2+
)deepfake_detector/reshape_for_rnn/ReshapeЉ
deepfake_detector/lstm/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   <    , 2
deepfake_detector/lstm/Shapeб
*deepfake_detector/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*deepfake_detector/lstm/strided_slice/stackд
,deepfake_detector/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,deepfake_detector/lstm/strided_slice/stack_1д
,deepfake_detector/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,deepfake_detector/lstm/strided_slice/stack_2В
$deepfake_detector/lstm/strided_sliceStridedSlice%deepfake_detector/lstm/Shape:output:03deepfake_detector/lstm/strided_slice/stack:output:05deepfake_detector/lstm/strided_slice/stack_1:output:05deepfake_detector/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$deepfake_detector/lstm/strided_sliceі
"deepfake_detector/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"deepfake_detector/lstm/zeros/mul/y╚
 deepfake_detector/lstm/zeros/mulMul-deepfake_detector/lstm/strided_slice:output:0+deepfake_detector/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2"
 deepfake_detector/lstm/zeros/mulЇ
#deepfake_detector/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2%
#deepfake_detector/lstm/zeros/Less/y├
!deepfake_detector/lstm/zeros/LessLess$deepfake_detector/lstm/zeros/mul:z:0,deepfake_detector/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2#
!deepfake_detector/lstm/zeros/Lessљ
%deepfake_detector/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%deepfake_detector/lstm/zeros/packed/1▀
#deepfake_detector/lstm/zeros/packedPack-deepfake_detector/lstm/strided_slice:output:0.deepfake_detector/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#deepfake_detector/lstm/zeros/packedЇ
"deepfake_detector/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"deepfake_detector/lstm/zeros/Const╚
deepfake_detector/lstm/zerosFill,deepfake_detector/lstm/zeros/packed:output:0+deepfake_detector/lstm/zeros/Const:output:0*
T0*
_output_shapes

:2
deepfake_detector/lstm/zerosј
$deepfake_detector/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$deepfake_detector/lstm/zeros_1/mul/y╬
"deepfake_detector/lstm/zeros_1/mulMul-deepfake_detector/lstm/strided_slice:output:0-deepfake_detector/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2$
"deepfake_detector/lstm/zeros_1/mulЉ
%deepfake_detector/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2'
%deepfake_detector/lstm/zeros_1/Less/y╦
#deepfake_detector/lstm/zeros_1/LessLess&deepfake_detector/lstm/zeros_1/mul:z:0.deepfake_detector/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2%
#deepfake_detector/lstm/zeros_1/Lessћ
'deepfake_detector/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'deepfake_detector/lstm/zeros_1/packed/1т
%deepfake_detector/lstm/zeros_1/packedPack-deepfake_detector/lstm/strided_slice:output:00deepfake_detector/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%deepfake_detector/lstm/zeros_1/packedЉ
$deepfake_detector/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$deepfake_detector/lstm/zeros_1/Constл
deepfake_detector/lstm/zeros_1Fill.deepfake_detector/lstm/zeros_1/packed:output:0-deepfake_detector/lstm/zeros_1/Const:output:0*
T0*
_output_shapes

:2 
deepfake_detector/lstm/zeros_1Б
%deepfake_detector/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%deepfake_detector/lstm/transpose/permС
 deepfake_detector/lstm/transpose	Transpose2deepfake_detector/reshape_for_rnn/Reshape:output:0.deepfake_detector/lstm/transpose/perm:output:0*
T0*$
_output_shapes
:<ап2"
 deepfake_detector/lstm/transposeЋ
deepfake_detector/lstm/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"<       , 2 
deepfake_detector/lstm/Shape_1д
,deepfake_detector/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,deepfake_detector/lstm/strided_slice_1/stackф
.deepfake_detector/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_1/stack_1ф
.deepfake_detector/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_1/stack_2Э
&deepfake_detector/lstm/strided_slice_1StridedSlice'deepfake_detector/lstm/Shape_1:output:05deepfake_detector/lstm/strided_slice_1/stack:output:07deepfake_detector/lstm/strided_slice_1/stack_1:output:07deepfake_detector/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_1│
2deepfake_detector/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         24
2deepfake_detector/lstm/TensorArrayV2/element_shapeј
$deepfake_detector/lstm/TensorArrayV2TensorListReserve;deepfake_detector/lstm/TensorArrayV2/element_shape:output:0/deepfake_detector/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$deepfake_detector/lstm/TensorArrayV2ь
Ldeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    , 2N
Ldeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeн
>deepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$deepfake_detector/lstm/transpose:y:0Udeepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>deepfake_detector/lstm/TensorArrayUnstack/TensorListFromTensorд
,deepfake_detector/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,deepfake_detector/lstm/strided_slice_2/stackф
.deepfake_detector/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_2/stack_1ф
.deepfake_detector/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_2/stack_2 
&deepfake_detector/lstm/strided_slice_2StridedSlice$deepfake_detector/lstm/transpose:y:05deepfake_detector/lstm/strided_slice_2/stack:output:07deepfake_detector/lstm/strided_slice_2/stack_1:output:07deepfake_detector/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ап*
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_2Ы
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp?deepfake_detector_lstm_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ап@*
dtype028
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOpШ
'deepfake_detector/lstm/lstm_cell/MatMulMatMul/deepfake_detector/lstm/strided_slice_2:output:0>deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@2)
'deepfake_detector/lstm/lstm_cell/MatMulШ
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpAdeepfake_detector_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02:
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOpЫ
)deepfake_detector/lstm/lstm_cell/MatMul_1MatMul%deepfake_detector/lstm/zeros:output:0@deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@2+
)deepfake_detector/lstm/lstm_cell/MatMul_1Т
$deepfake_detector/lstm/lstm_cell/addAddV21deepfake_detector/lstm/lstm_cell/MatMul:product:03deepfake_detector/lstm/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

:@2&
$deepfake_detector/lstm/lstm_cell/add№
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp@deepfake_detector_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOpз
(deepfake_detector/lstm/lstm_cell/BiasAddBiasAdd(deepfake_detector/lstm/lstm_cell/add:z:0?deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@2*
(deepfake_detector/lstm/lstm_cell/BiasAddд
0deepfake_detector/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0deepfake_detector/lstm/lstm_cell/split/split_dimЪ
&deepfake_detector/lstm/lstm_cell/splitSplit9deepfake_detector/lstm/lstm_cell/split/split_dim:output:01deepfake_detector/lstm/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split2(
&deepfake_detector/lstm/lstm_cell/split╣
(deepfake_detector/lstm/lstm_cell/SigmoidSigmoid/deepfake_detector/lstm/lstm_cell/split:output:0*
T0*
_output_shapes

:2*
(deepfake_detector/lstm/lstm_cell/Sigmoidй
*deepfake_detector/lstm/lstm_cell/Sigmoid_1Sigmoid/deepfake_detector/lstm/lstm_cell/split:output:1*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/lstm_cell/Sigmoid_1Н
$deepfake_detector/lstm/lstm_cell/mulMul.deepfake_detector/lstm/lstm_cell/Sigmoid_1:y:0'deepfake_detector/lstm/zeros_1:output:0*
T0*
_output_shapes

:2&
$deepfake_detector/lstm/lstm_cell/mul░
%deepfake_detector/lstm/lstm_cell/ReluRelu/deepfake_detector/lstm/lstm_cell/split:output:2*
T0*
_output_shapes

:2'
%deepfake_detector/lstm/lstm_cell/Reluс
&deepfake_detector/lstm/lstm_cell/mul_1Mul,deepfake_detector/lstm/lstm_cell/Sigmoid:y:03deepfake_detector/lstm/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/mul_1п
&deepfake_detector/lstm/lstm_cell/add_1AddV2(deepfake_detector/lstm/lstm_cell/mul:z:0*deepfake_detector/lstm/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/add_1й
*deepfake_detector/lstm/lstm_cell/Sigmoid_2Sigmoid/deepfake_detector/lstm/lstm_cell/split:output:3*
T0*
_output_shapes

:2,
*deepfake_detector/lstm/lstm_cell/Sigmoid_2»
'deepfake_detector/lstm/lstm_cell/Relu_1Relu*deepfake_detector/lstm/lstm_cell/add_1:z:0*
T0*
_output_shapes

:2)
'deepfake_detector/lstm/lstm_cell/Relu_1у
&deepfake_detector/lstm/lstm_cell/mul_2Mul.deepfake_detector/lstm/lstm_cell/Sigmoid_2:y:05deepfake_detector/lstm/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:2(
&deepfake_detector/lstm/lstm_cell/mul_2й
4deepfake_detector/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      26
4deepfake_detector/lstm/TensorArrayV2_1/element_shapeћ
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
deepfake_detector/lstm/timeГ
/deepfake_detector/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         21
/deepfake_detector/lstm/while/maximum_iterationsў
)deepfake_detector/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2+
)deepfake_detector/lstm/while/loop_counter╣
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
deepfake_detector/lstm/whileс
Gdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2I
Gdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shape╗
9deepfake_detector/lstm/TensorArrayV2Stack/TensorListStackTensorListStack%deepfake_detector/lstm/while:output:3Pdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:<*
element_dtype02;
9deepfake_detector/lstm/TensorArrayV2Stack/TensorListStack»
,deepfake_detector/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2.
,deepfake_detector/lstm/strided_slice_3/stackф
.deepfake_detector/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.deepfake_detector/lstm/strided_slice_3/stack_1ф
.deepfake_detector/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.deepfake_detector/lstm/strided_slice_3/stack_2Џ
&deepfake_detector/lstm/strided_slice_3StridedSliceBdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack:tensor:05deepfake_detector/lstm/strided_slice_3/stack:output:07deepfake_detector/lstm/strided_slice_3/stack_1:output:07deepfake_detector/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&deepfake_detector/lstm/strided_slice_3Д
'deepfake_detector/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'deepfake_detector/lstm/transpose_1/permЭ
"deepfake_detector/lstm/transpose_1	TransposeBdeepfake_detector/lstm/TensorArrayV2Stack/TensorListStack:tensor:00deepfake_detector/lstm/transpose_1/perm:output:0*
T0*"
_output_shapes
:<2$
"deepfake_detector/lstm/transpose_1ћ
deepfake_detector/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2 
deepfake_detector/lstm/runtimeЊ
deepfake_detector/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
deepfake_detector/flatten/ConstН
!deepfake_detector/flatten/ReshapeReshape/deepfake_detector/lstm/strided_slice_3:output:0(deepfake_detector/flatten/Const:output:0*
T0*
_output_shapes

:2#
!deepfake_detector/flatten/ReshapeН
-deepfake_detector/dense/MatMul/ReadVariableOpReadVariableOp6deepfake_detector_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-deepfake_detector/dense/MatMul/ReadVariableOpо
deepfake_detector/dense/MatMulMatMul*deepfake_detector/flatten/Reshape:output:05deepfake_detector/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2 
deepfake_detector/dense/MatMulн
.deepfake_detector/dense/BiasAdd/ReadVariableOpReadVariableOp7deepfake_detector_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.deepfake_detector/dense/BiasAdd/ReadVariableOpп
deepfake_detector/dense/BiasAddBiasAdd(deepfake_detector/dense/MatMul:product:06deepfake_detector/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2!
deepfake_detector/dense/BiasAddа
deepfake_detector/dense/SigmoidSigmoid(deepfake_detector/dense/BiasAdd:output:0*
T0*
_output_shapes

:2!
deepfake_detector/dense/Sigmoidu
IdentityIdentity#deepfake_detector/dense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

IdentityЂ
NoOpNoOp/^deepfake_detector/dense/BiasAdd/ReadVariableOp.^deepfake_detector/dense/MatMul/ReadVariableOp8^deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp7^deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp9^deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp^deepfake_detector/lstm/whileA^deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp@^deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         <dd: : : : : : : 2`
.deepfake_detector/dense/BiasAdd/ReadVariableOp.deepfake_detector/dense/BiasAdd/ReadVariableOp2^
-deepfake_detector/dense/MatMul/ReadVariableOp-deepfake_detector/dense/MatMul/ReadVariableOp2r
7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp7deepfake_detector/lstm/lstm_cell/BiasAdd/ReadVariableOp2p
6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp6deepfake_detector/lstm/lstm_cell/MatMul/ReadVariableOp2t
8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp8deepfake_detector/lstm/lstm_cell/MatMul_1/ReadVariableOp2<
deepfake_detector/lstm/whiledeepfake_detector/lstm/while2ё
@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp@deepfake_detector/time_distributed/conv2d/BiasAdd/ReadVariableOp2ѓ
?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp?deepfake_detector/time_distributed/conv2d/Conv2D/ReadVariableOp:\ X
3
_output_shapes!
:         <dd
!
_user_specified_name	input_1
ф
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
strided_slice/stack_2Р
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
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
ReshapeЁ
!average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         11 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_74197122#
!average_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape«
	Reshape_1Reshape*average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs
Т
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
strided_slice/stack_2Р
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
valueB"    c   c       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         cc 2	
Reshape╔
average_pooling2d/AvgPoolAvgPoolReshape:output:0*
T0*/
_output_shapes
:         11 *
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
         2
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
Reshape_1/shape/4Я
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeд
	Reshape_1Reshape"average_pooling2d/AvgPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&                  11 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&                  11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&                  cc :d `
<
_output_shapes*
(:&                  cc 
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_defaultџ
G
input_1<
serving_default_input_1:0         <dd3
output_1'
StatefulPartitionedCall:0tensorflow/serving/predict: ├
џ
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
░
	layer
	variables
regularization_losses
trainable_variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_layer
░
	layer
	variables
regularization_losses
trainable_variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
Ц
	variables
regularization_losses
trainable_variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"
_tf_keras_layer
├
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
Ц
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*z&call_and_return_all_conditional_losses
{__call__"
_tf_keras_layer
╗

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
╩
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
╝

*kernel
+bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
*&call_and_return_all_conditional_losses
ђ__call__"
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
Г
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
Д
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+Ђ&call_and_return_all_conditional_losses
ѓ__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
Г
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
с
K
state_size

,kernel
-recurrent_kernel
.bias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+Ѓ&call_and_return_all_conditional_losses
ё__call__"
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
╣
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
Г
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
Г
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
ап@ 2'deepfake_detector/lstm/lstm_cell/kernel
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
«
`non_trainable_variables

alayers
blayer_metrics
4	variables
cmetrics
dlayer_regularization_losses
5regularization_losses
6trainable_variables
ђ__call__
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
░
enon_trainable_variables

flayers
glayer_metrics
=	variables
hmetrics
ilayer_regularization_losses
>regularization_losses
?trainable_variables
ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
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
░
jnon_trainable_variables

klayers
llayer_metrics
L	variables
mmetrics
nlayer_regularization_losses
Mregularization_losses
Ntrainable_variables
ё__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
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
═B╩
"__inference__wrapped_model_7419524input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щ2Ш
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ї2і
3__inference_deepfake_detector_layer_call_fn_7421834
3__inference_deepfake_detector_layer_call_fn_7421853
3__inference_deepfake_detector_layer_call_fn_7421872
3__inference_deepfake_detector_layer_call_fn_7421891│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓ2 
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
2__inference_time_distributed_layer_call_fn_7421978
2__inference_time_distributed_layer_call_fn_7421987
2__inference_time_distributed_layer_call_fn_7421996
2__inference_time_distributed_layer_call_fn_7422005└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422023
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422041
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
4__inference_time_distributed_1_layer_call_fn_7422064
4__inference_time_distributed_1_layer_call_fn_7422069
4__inference_time_distributed_1_layer_call_fn_7422074
4__inference_time_distributed_1_layer_call_fn_7422079└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ш2з
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7422085б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
█2п
1__inference_reshape_for_rnn_layer_call_fn_7422090б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
у2С
A__inference_lstm_layer_call_and_return_conditional_losses_7422241
A__inference_lstm_layer_call_and_return_conditional_losses_7422392
A__inference_lstm_layer_call_and_return_conditional_losses_7422543
A__inference_lstm_layer_call_and_return_conditional_losses_7422694Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ч2Э
&__inference_lstm_layer_call_fn_7422705
&__inference_lstm_layer_call_fn_7422716
&__inference_lstm_layer_call_fn_7422727
&__inference_lstm_layer_call_fn_7422738Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_flatten_layer_call_and_return_conditional_losses_7422744б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_flatten_layer_call_fn_7422749б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_layer_call_and_return_conditional_losses_7422760б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_layer_call_fn_7422769б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠B╔
%__inference_signature_wrapper_7421087input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv2d_layer_call_fn_7422789б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њ2Ј
3__inference_average_pooling2d_layer_call_fn_7422804
3__inference_average_pooling2d_layer_call_fn_7422809б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
+__inference_lstm_cell_layer_call_fn_7422890
+__inference_lstm_cell_layer_call_fn_7422907Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 Ў
"__inference__wrapped_model_7419524s*+,-.$%<б9
2б/
-і*
input_1         <dd
ф "*ф'
%
output_1і
output_1ы
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422794ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ║
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7422799h7б4
-б*
(і%
inputs         cc 
ф "-б*
#і 
0         11 
џ ╔
3__inference_average_pooling2d_layer_call_fn_7422804ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    њ
3__inference_average_pooling2d_layer_call_fn_7422809[7б4
-б*
(і%
inputs         cc 
ф " і         11 │
C__inference_conv2d_layer_call_and_return_conditional_losses_7422780l*+7б4
-б*
(і%
inputs         dd
ф "-б*
#і 
0         cc 
џ І
(__inference_conv2d_layer_call_fn_7422789_*+7б4
-б*
(і%
inputs         dd
ф " і         cc ║
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421269h*+,-.$%?б<
5б2
,і)
inputs         <dd
p 
ф "б
і
0
џ ║
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421451h*+,-.$%?б<
5б2
,і)
inputs         <dd
p
ф "б
і
0
џ ╗
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421633i*+,-.$%@б=
6б3
-і*
input_1         <dd
p 
ф "б
і
0
џ ╗
N__inference_deepfake_detector_layer_call_and_return_conditional_losses_7421815i*+,-.$%@б=
6б3
-і*
input_1         <dd
p
ф "б
і
0
џ Њ
3__inference_deepfake_detector_layer_call_fn_7421834\*+,-.$%@б=
6б3
-і*
input_1         <dd
p 
ф "іњ
3__inference_deepfake_detector_layer_call_fn_7421853[*+,-.$%?б<
5б2
,і)
inputs         <dd
p 
ф "іњ
3__inference_deepfake_detector_layer_call_fn_7421872[*+,-.$%?б<
5б2
,і)
inputs         <dd
p
ф "іЊ
3__inference_deepfake_detector_layer_call_fn_7421891\*+,-.$%@б=
6б3
-і*
input_1         <dd
p
ф "іљ
B__inference_dense_layer_call_and_return_conditional_losses_7422760J$%&б#
б
і
inputs
ф "б
і
0
џ h
'__inference_dense_layer_call_fn_7422769=$%&б#
б
і
inputs
ф "іј
D__inference_flatten_layer_call_and_return_conditional_losses_7422744F&б#
б
і
inputs
ф "б
і
0
џ f
)__inference_flatten_layer_call_fn_74227499&б#
б
і
inputs
ф "і╩
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422841 ,-.ѓб
xбu
"і
inputs         ап
KбH
"і
states/0         
"і
states/1         
p 
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ ╩
F__inference_lstm_cell_layer_call_and_return_conditional_losses_7422873 ,-.ѓб
xбu
"і
inputs         ап
KбH
"і
states/0         
"і
states/1         
p
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ Ъ
+__inference_lstm_cell_layer_call_fn_7422890№,-.ѓб
xбu
"і
inputs         ап
KбH
"і
states/0         
"і
states/1         
p 
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         Ъ
+__inference_lstm_cell_layer_call_fn_7422907№,-.ѓб
xбu
"і
inputs         ап
KбH
"і
states/0         
"і
states/1         
p
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         ─
A__inference_lstm_layer_call_and_return_conditional_losses_7422241,-.QбN
GбD
6џ3
1і.
inputs/0                   ап

 
p 

 
ф "%б"
і
0         
џ ─
A__inference_lstm_layer_call_and_return_conditional_losses_7422392,-.QбN
GбD
6џ3
1і.
inputs/0                   ап

 
p

 
ф "%б"
і
0         
џ б
A__inference_lstm_layer_call_and_return_conditional_losses_7422543],-.8б5
.б+
і
inputs<ап

 
p 

 
ф "б
і
0
џ б
A__inference_lstm_layer_call_and_return_conditional_losses_7422694],-.8б5
.б+
і
inputs<ап

 
p

 
ф "б
і
0
џ ю
&__inference_lstm_layer_call_fn_7422705r,-.QбN
GбD
6џ3
1і.
inputs/0                   ап

 
p 

 
ф "і         ю
&__inference_lstm_layer_call_fn_7422716r,-.QбN
GбD
6џ3
1і.
inputs/0                   ап

 
p

 
ф "і         z
&__inference_lstm_layer_call_fn_7422727P,-.8б5
.б+
і
inputs<ап

 
p 

 
ф "іz
&__inference_lstm_layer_call_fn_7422738P,-.8б5
.б+
і
inputs<ап

 
p

 
ф "і▒
L__inference_reshape_for_rnn_layer_call_and_return_conditional_losses_7422085a;б8
1б.
,і)
inputs         <11 
ф ""б
і
0<ап
џ Ѕ
1__inference_reshape_for_rnn_layer_call_fn_7422090T;б8
1б.
,і)
inputs         <11 
ф "і<апД
%__inference_signature_wrapper_7421087~*+,-.$%GбD
б 
=ф:
8
input_1-і*
input_1         <dd"*ф'
%
output_1і
output_1я
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422023іLбI
Bб?
5і2
inputs&                  cc 
p 

 
ф ":б7
0і-
0&                  11 
џ я
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422041іLбI
Bб?
5і2
inputs&                  cc 
p

 
ф ":б7
0і-
0&                  11 
џ ╦
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422050xCб@
9б6
,і)
inputs         <cc 
p 

 
ф "1б.
'і$
0         <11 
џ ╦
O__inference_time_distributed_1_layer_call_and_return_conditional_losses_7422059xCб@
9б6
,і)
inputs         <cc 
p

 
ф "1б.
'і$
0         <11 
џ х
4__inference_time_distributed_1_layer_call_fn_7422064}LбI
Bб?
5і2
inputs&                  cc 
p 

 
ф "-і*&                  11 х
4__inference_time_distributed_1_layer_call_fn_7422069}LбI
Bб?
5і2
inputs&                  cc 
p

 
ф "-і*&                  11 Б
4__inference_time_distributed_1_layer_call_fn_7422074kCб@
9б6
,і)
inputs         <cc 
p 

 
ф "$і!         <11 Б
4__inference_time_distributed_1_layer_call_fn_7422079kCб@
9б6
,і)
inputs         <cc 
p

 
ф "$і!         <11 Я
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421915ј*+LбI
Bб?
5і2
inputs&                  dd
p 

 
ф ":б7
0і-
0&                  cc 
џ Я
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421939ј*+LбI
Bб?
5і2
inputs&                  dd
p

 
ф ":б7
0і-
0&                  cc 
џ ═
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421954|*+Cб@
9б6
,і)
inputs         <dd
p 

 
ф "1б.
'і$
0         <cc 
џ ═
M__inference_time_distributed_layer_call_and_return_conditional_losses_7421969|*+Cб@
9б6
,і)
inputs         <dd
p

 
ф "1б.
'і$
0         <cc 
џ И
2__inference_time_distributed_layer_call_fn_7421978Ђ*+LбI
Bб?
5і2
inputs&                  dd
p 

 
ф "-і*&                  cc И
2__inference_time_distributed_layer_call_fn_7421987Ђ*+LбI
Bб?
5і2
inputs&                  dd
p

 
ф "-і*&                  cc Ц
2__inference_time_distributed_layer_call_fn_7421996o*+Cб@
9б6
,і)
inputs         <dd
p 

 
ф "$і!         <cc Ц
2__inference_time_distributed_layer_call_fn_7422005o*+Cб@
9б6
,і)
inputs         <dd
p

 
ф "$і!         <cc 