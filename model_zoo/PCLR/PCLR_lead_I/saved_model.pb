ЕЪ9
╤г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878╔└/
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:@*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
:@А*
dtype0
П
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:А*
dtype0
А
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv1d_3/kernel
y
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*$
_output_shapes
:АА*
dtype0

conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv1d_1/kernel
x
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*#
_output_shapes
:@А*
dtype0
П
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_2/gamma
И
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_2/beta
Ж
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_2/moving_mean
Ф
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_2/moving_variance
Ь
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:А*
dtype0
А
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А─* 
shared_nameconv1d_5/kernel
y
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*$
_output_shapes
:А─*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:─*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:─*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:─*
dtype0
г
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:─*
dtype0
А
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:──* 
shared_nameconv1d_6/kernel
y
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*$
_output_shapes
:──*
dtype0
А
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А─* 
shared_nameconv1d_4/kernel
y
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*$
_output_shapes
:А─*
dtype0
П
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*,
shared_namebatch_normalization_4/gamma
И
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:─*
dtype0
Н
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*+
shared_namebatch_normalization_4/beta
Ж
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:─*
dtype0
Ы
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*2
shared_name#!batch_normalization_4/moving_mean
Ф
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:─*
dtype0
г
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*6
shared_name'%batch_normalization_4/moving_variance
Ь
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:─*
dtype0
А
conv1d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:─А* 
shared_nameconv1d_8/kernel
y
#conv1d_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_8/kernel*$
_output_shapes
:─А*
dtype0
П
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_5/gamma
И
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_5/beta
Ж
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_5/moving_mean
Ф
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_5/moving_variance
Ь
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:А*
dtype0
А
conv1d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv1d_9/kernel
y
#conv1d_9/kernel/Read/ReadVariableOpReadVariableOpconv1d_9/kernel*$
_output_shapes
:АА*
dtype0
А
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:─А* 
shared_nameconv1d_7/kernel
y
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*$
_output_shapes
:─А*
dtype0
П
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_6/gamma
И
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_6/beta
Ж
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_6/moving_mean
Ф
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_6/moving_variance
Ь
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:А*
dtype0
В
conv1d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А└*!
shared_nameconv1d_11/kernel
{
$conv1d_11/kernel/Read/ReadVariableOpReadVariableOpconv1d_11/kernel*$
_output_shapes
:А└*
dtype0
П
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*,
shared_namebatch_normalization_7/gamma
И
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:└*
dtype0
Н
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*+
shared_namebatch_normalization_7/beta
Ж
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:└*
dtype0
Ы
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*2
shared_name#!batch_normalization_7/moving_mean
Ф
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:└*
dtype0
г
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*6
shared_name'%batch_normalization_7/moving_variance
Ь
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:└*
dtype0
В
conv1d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:└└*!
shared_nameconv1d_12/kernel
{
$conv1d_12/kernel/Read/ReadVariableOpReadVariableOpconv1d_12/kernel*$
_output_shapes
:└└*
dtype0
В
conv1d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А└*!
shared_nameconv1d_10/kernel
{
$conv1d_10/kernel/Read/ReadVariableOpReadVariableOpconv1d_10/kernel*$
_output_shapes
:А└*
dtype0
П
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*,
shared_namebatch_normalization_8/gamma
И
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:└*
dtype0
Н
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*+
shared_namebatch_normalization_8/beta
Ж
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:└*
dtype0
Ы
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*2
shared_name#!batch_normalization_8/moving_mean
Ф
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:└*
dtype0
г
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*6
shared_name'%batch_normalization_8/moving_variance
Ь
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:└*
dtype0

NoOpNoOp
╖г
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ёв
valueцвBтв B┌в
▐	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer_with_weights-16
layer-29
layer-30
 layer_with_weights-17
 layer-31
!layer_with_weights-18
!layer-32
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer_with_weights-20
%layer-36
&layer-37
'layer_with_weights-21
'layer-38
(layer-39
)layer-40
*trainable_variables
+regularization_losses
,	variables
-	keras_api
.
signatures
 
^

/kernel
0trainable_variables
1regularization_losses
2	variables
3	keras_api
Ч
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
^

Akernel
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
Ч
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
R
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
^

Wkernel
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
^

\kernel
]trainable_variables
^regularization_losses
_	variables
`	keras_api
R
atrainable_variables
bregularization_losses
c	variables
d	keras_api
Ч
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
R
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
^

rkernel
strainable_variables
tregularization_losses
u	variables
v	keras_api
Ч
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
V
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
V
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
c
Иkernel
Йtrainable_variables
Кregularization_losses
Л	variables
М	keras_api
c
Нkernel
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
V
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
а
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ыtrainable_variables
Ьregularization_losses
Э	variables
Ю	keras_api
V
Яtrainable_variables
аregularization_losses
б	variables
в	keras_api
c
гkernel
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
а
	иaxis

йgamma
	кbeta
лmoving_mean
мmoving_variance
нtrainable_variables
оregularization_losses
п	variables
░	keras_api
V
▒trainable_variables
▓regularization_losses
│	variables
┤	keras_api
V
╡trainable_variables
╢regularization_losses
╖	variables
╕	keras_api
c
╣kernel
║trainable_variables
╗regularization_losses
╝	variables
╜	keras_api
c
╛kernel
┐trainable_variables
└regularization_losses
┴	variables
┬	keras_api
V
├trainable_variables
─regularization_losses
┼	variables
╞	keras_api
а
	╟axis

╚gamma
	╔beta
╩moving_mean
╦moving_variance
╠trainable_variables
═regularization_losses
╬	variables
╧	keras_api
V
╨trainable_variables
╤regularization_losses
╥	variables
╙	keras_api
c
╘kernel
╒trainable_variables
╓regularization_losses
╫	variables
╪	keras_api
а
	┘axis

┌gamma
	█beta
▄moving_mean
▌moving_variance
▐trainable_variables
▀regularization_losses
р	variables
с	keras_api
V
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
V
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
c
ъkernel
ыtrainable_variables
ьregularization_losses
э	variables
ю	keras_api
c
яkernel
Ёtrainable_variables
ёregularization_losses
Є	variables
є	keras_api
V
Їtrainable_variables
їregularization_losses
Ў	variables
ў	keras_api
а
	°axis

∙gamma
	·beta
√moving_mean
№moving_variance
¤trainable_variables
■regularization_losses
 	variables
А	keras_api
V
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
V
Еtrainable_variables
Жregularization_losses
З	variables
И	keras_api
А
/0
51
62
A3
G4
H5
W6
\7
f8
g9
r10
x11
y12
И13
Н14
Ч15
Ш16
г17
й18
к19
╣20
╛21
╚22
╔23
╘24
┌25
█26
ъ27
я28
∙29
·30
 
Ъ
/0
51
62
73
84
A5
G6
H7
I8
J9
W10
\11
f12
g13
h14
i15
r16
x17
y18
z19
{20
И21
Н22
Ч23
Ш24
Щ25
Ъ26
г27
й28
к29
л30
м31
╣32
╛33
╚34
╔35
╩36
╦37
╘38
┌39
█40
▄41
▌42
ъ43
я44
∙45
·46
√47
№48
▓
*trainable_variables
+regularization_losses
 Йlayer_regularization_losses
,	variables
Кmetrics
Лlayer_metrics
Мlayers
Нnon_trainable_variables
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

/0
 

/0
▓
0trainable_variables
 Оlayer_regularization_losses
1regularization_losses
2	variables
Пmetrics
Рlayer_metrics
Сlayers
Тnon_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
72
83
▓
9trainable_variables
 Уlayer_regularization_losses
:regularization_losses
;	variables
Фmetrics
Хlayer_metrics
Цlayers
Чnon_trainable_variables
 
 
 
▓
=trainable_variables
 Шlayer_regularization_losses
>regularization_losses
?	variables
Щmetrics
Ъlayer_metrics
Ыlayers
Ьnon_trainable_variables
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

A0
 

A0
▓
Btrainable_variables
 Эlayer_regularization_losses
Cregularization_losses
D	variables
Юmetrics
Яlayer_metrics
аlayers
бnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
I2
J3
▓
Ktrainable_variables
 вlayer_regularization_losses
Lregularization_losses
M	variables
гmetrics
дlayer_metrics
еlayers
жnon_trainable_variables
 
 
 
▓
Otrainable_variables
 зlayer_regularization_losses
Pregularization_losses
Q	variables
иmetrics
йlayer_metrics
кlayers
лnon_trainable_variables
 
 
 
▓
Strainable_variables
 мlayer_regularization_losses
Tregularization_losses
U	variables
нmetrics
оlayer_metrics
пlayers
░non_trainable_variables
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

W0
 

W0
▓
Xtrainable_variables
 ▒layer_regularization_losses
Yregularization_losses
Z	variables
▓metrics
│layer_metrics
┤layers
╡non_trainable_variables
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE

\0
 

\0
▓
]trainable_variables
 ╢layer_regularization_losses
^regularization_losses
_	variables
╖metrics
╕layer_metrics
╣layers
║non_trainable_variables
 
 
 
▓
atrainable_variables
 ╗layer_regularization_losses
bregularization_losses
c	variables
╝metrics
╜layer_metrics
╛layers
┐non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
h2
i3
▓
jtrainable_variables
 └layer_regularization_losses
kregularization_losses
l	variables
┴metrics
┬layer_metrics
├layers
─non_trainable_variables
 
 
 
▓
ntrainable_variables
 ┼layer_regularization_losses
oregularization_losses
p	variables
╞metrics
╟layer_metrics
╚layers
╔non_trainable_variables
[Y
VARIABLE_VALUEconv1d_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE

r0
 

r0
▓
strainable_variables
 ╩layer_regularization_losses
tregularization_losses
u	variables
╦metrics
╠layer_metrics
═layers
╬non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
z2
{3
▓
|trainable_variables
 ╧layer_regularization_losses
}regularization_losses
~	variables
╨metrics
╤layer_metrics
╥layers
╙non_trainable_variables
 
 
 
╡
Аtrainable_variables
 ╘layer_regularization_losses
Бregularization_losses
В	variables
╒metrics
╓layer_metrics
╫layers
╪non_trainable_variables
 
 
 
╡
Дtrainable_variables
 ┘layer_regularization_losses
Еregularization_losses
Ж	variables
┌metrics
█layer_metrics
▄layers
▌non_trainable_variables
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE

И0
 

И0
╡
Йtrainable_variables
 ▐layer_regularization_losses
Кregularization_losses
Л	variables
▀metrics
рlayer_metrics
сlayers
тnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE

Н0
 

Н0
╡
Оtrainable_variables
 уlayer_regularization_losses
Пregularization_losses
Р	variables
фmetrics
хlayer_metrics
цlayers
чnon_trainable_variables
 
 
 
╡
Тtrainable_variables
 шlayer_regularization_losses
Уregularization_losses
Ф	variables
щmetrics
ъlayer_metrics
ыlayers
ьnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_4/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_4/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_4/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_4/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Ч0
Ш1
 
 
Ч0
Ш1
Щ2
Ъ3
╡
Ыtrainable_variables
 эlayer_regularization_losses
Ьregularization_losses
Э	variables
юmetrics
яlayer_metrics
Ёlayers
ёnon_trainable_variables
 
 
 
╡
Яtrainable_variables
 Єlayer_regularization_losses
аregularization_losses
б	variables
єmetrics
Їlayer_metrics
їlayers
Ўnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_8/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE

г0
 

г0
╡
дtrainable_variables
 ўlayer_regularization_losses
еregularization_losses
ж	variables
°metrics
∙layer_metrics
·layers
√non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

й0
к1
 
 
й0
к1
л2
м3
╡
нtrainable_variables
 №layer_regularization_losses
оregularization_losses
п	variables
¤metrics
■layer_metrics
 layers
Аnon_trainable_variables
 
 
 
╡
▒trainable_variables
 Бlayer_regularization_losses
▓regularization_losses
│	variables
Вmetrics
Гlayer_metrics
Дlayers
Еnon_trainable_variables
 
 
 
╡
╡trainable_variables
 Жlayer_regularization_losses
╢regularization_losses
╖	variables
Зmetrics
Иlayer_metrics
Йlayers
Кnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_9/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE

╣0
 

╣0
╡
║trainable_variables
 Лlayer_regularization_losses
╗regularization_losses
╝	variables
Мmetrics
Нlayer_metrics
Оlayers
Пnon_trainable_variables
\Z
VARIABLE_VALUEconv1d_7/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE

╛0
 

╛0
╡
┐trainable_variables
 Рlayer_regularization_losses
└regularization_losses
┴	variables
Сmetrics
Тlayer_metrics
Уlayers
Фnon_trainable_variables
 
 
 
╡
├trainable_variables
 Хlayer_regularization_losses
─regularization_losses
┼	variables
Цmetrics
Чlayer_metrics
Шlayers
Щnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

╚0
╔1
 
 
╚0
╔1
╩2
╦3
╡
╠trainable_variables
 Ъlayer_regularization_losses
═regularization_losses
╬	variables
Ыmetrics
Ьlayer_metrics
Эlayers
Юnon_trainable_variables
 
 
 
╡
╨trainable_variables
 Яlayer_regularization_losses
╤regularization_losses
╥	variables
аmetrics
бlayer_metrics
вlayers
гnon_trainable_variables
][
VARIABLE_VALUEconv1d_11/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE

╘0
 

╘0
╡
╒trainable_variables
 дlayer_regularization_losses
╓regularization_losses
╫	variables
еmetrics
жlayer_metrics
зlayers
иnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

┌0
█1
 
 
┌0
█1
▄2
▌3
╡
▐trainable_variables
 йlayer_regularization_losses
▀regularization_losses
р	variables
кmetrics
лlayer_metrics
мlayers
нnon_trainable_variables
 
 
 
╡
тtrainable_variables
 оlayer_regularization_losses
уregularization_losses
ф	variables
пmetrics
░layer_metrics
▒layers
▓non_trainable_variables
 
 
 
╡
цtrainable_variables
 │layer_regularization_losses
чregularization_losses
ш	variables
┤metrics
╡layer_metrics
╢layers
╖non_trainable_variables
][
VARIABLE_VALUEconv1d_12/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE

ъ0
 

ъ0
╡
ыtrainable_variables
 ╕layer_regularization_losses
ьregularization_losses
э	variables
╣metrics
║layer_metrics
╗layers
╝non_trainable_variables
][
VARIABLE_VALUEconv1d_10/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE

я0
 

я0
╡
Ёtrainable_variables
 ╜layer_regularization_losses
ёregularization_losses
Є	variables
╛metrics
┐layer_metrics
└layers
┴non_trainable_variables
 
 
 
╡
Їtrainable_variables
 ┬layer_regularization_losses
їregularization_losses
Ў	variables
├metrics
─layer_metrics
┼layers
╞non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

∙0
·1
 
 
∙0
·1
√2
№3
╡
¤trainable_variables
 ╟layer_regularization_losses
■regularization_losses
 	variables
╚metrics
╔layer_metrics
╩layers
╦non_trainable_variables
 
 
 
╡
Бtrainable_variables
 ╠layer_regularization_losses
Вregularization_losses
Г	variables
═metrics
╬layer_metrics
╧layers
╨non_trainable_variables
 
 
 
╡
Еtrainable_variables
 ╤layer_regularization_losses
Жregularization_losses
З	variables
╥metrics
╙layer_metrics
╘layers
╒non_trainable_variables
 
 
 
╛
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
Р
70
81
I2
J3
h4
i5
z6
{7
Щ8
Ъ9
л10
м11
╩12
╦13
▄14
▌15
√16
№17
 
 
 
 
 
 
 
 
 

70
81
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

I0
J1
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

h0
i1
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

z0
{1
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

Щ0
Ъ1
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

л0
м1
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

╩0
╦1
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

▄0
▌1
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

√0
№1
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
А
serving_default_ecgPlaceholder*,
_output_shapes
:         А *
dtype0*!
shape:         А 
А
StatefulPartitionedCallStatefulPartitionedCallserving_default_ecgconv1d/kernel#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv1d_2/kernel%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv1d_3/kernelconv1d_1/kernel%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betaconv1d_5/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betaconv1d_6/kernelconv1d_4/kernel%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaconv1d_8/kernel%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betaconv1d_9/kernelconv1d_7/kernel%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/betaconv1d_11/kernel%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/betaconv1d_12/kernelconv1d_10/kernel%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/beta*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./01*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_4850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
П
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv1d_6/kernel/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv1d_8/kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv1d_9/kernel/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp$conv1d_11/kernel/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$conv1d_12/kernel/Read/ReadVariableOp$conv1d_10/kernel/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOpConst*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_7819
╢
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv1d_2/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv1d_3/kernelconv1d_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv1d_5/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv1d_6/kernelconv1d_4/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv1d_8/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv1d_9/kernelconv1d_7/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv1d_11/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv1d_12/kernelconv1d_10/kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_7976ь№,
щ
з
4__inference_batch_normalization_3_layer_call_fn_6464

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19752
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
В
У
C__inference_conv1d_12_layer_call_and_return_conditional_losses_3944

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @└2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:└└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:└└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @└::T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╛№
м
F__inference_functional_1_layer_call_and_return_conditional_losses_5560

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource8
4conv1d_8_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource8
4conv1d_9_conv1d_expanddims_1_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resource9
5conv1d_12_conv1d_expanddims_1_readvariableop_resource9
5conv1d_10_conv1d_expanddims_1_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource?
;batch_normalization_8_batchnorm_mul_readvariableop_resource=
9batch_normalization_8_batchnorm_readvariableop_1_resource=
9batch_normalization_8_batchnorm_readvariableop_2_resource
identityИЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimм
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А 2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/conv1d/ExpandDims_1╙
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         А @*
paddingSAME*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         А @*
squeeze_dims

¤        2
conv1d/conv1d/Squeeze╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╪
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul╧
#batch_normalization/batchnorm/mul_1Mulconv1d/conv1d/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         А @2
activation/ReluЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╔
conv1d_2/conv1d/ExpandDims
ExpandDimsactivation/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
conv1d_2/conv1d/ExpandDims╘
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim▄
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А А*
paddingSAME*
strides
2
conv1d_2/conv1dп
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*-
_output_shapes
:         А А*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeeze╒
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yс
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/addж
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_1/batchnorm/Rsqrtс
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▐
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/mul╪
%batch_normalization_1/batchnorm/mul_1Mul conv1d_2/conv1d/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2'
%batch_normalization_1/batchnorm/mul_1█
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▐
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_1/batchnorm/mul_2█
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2▄
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/subу
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2'
%batch_normalization_1/batchnorm/add_1~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim├
max_pooling1d/ExpandDims
ExpandDimsactivation/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
max_pooling1d/ExpandDims╔
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:         А@*
ksize
*
paddingSAME*
strides
2
max_pooling1d/MaxPoolз
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:         А@*
squeeze_dims
2
max_pooling1d/SqueezeС
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А А2
activation_1/ReluЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_3/conv1d/ExpandDims/dim╠
conv1d_3/conv1d/ExpandDims
ExpandDimsactivation_1/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А А2
conv1d_3/conv1d/ExpandDims╒
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim▌
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d_3/conv1d/ExpandDims_1▄
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_3/conv1dп
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_3/conv1d/SqueezeЛ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╩
conv1d_1/conv1d/ExpandDims
ExpandDimsmax_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А@2
conv1d_1/conv1d/ExpandDims╘
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim▄
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_1/conv1dп
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_1/conv1d/SqueezeЧ
add/addAddV2 conv1d_3/conv1d/Squeeze:output:0 conv1d_1/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2	
add/add╒
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yс
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/addж
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_2/batchnorm/Rsqrtс
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▐
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/mul├
%batch_normalization_2/batchnorm/mul_1Muladd/add:z:0'batch_normalization_2/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_2/batchnorm/mul_1█
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▐
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_2/batchnorm/mul_2█
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2▄
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/subу
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_2/batchnorm/add_1С
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2
activation_2/ReluЛ
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_5/conv1d/ExpandDims/dim╠
conv1d_5/conv1d/ExpandDims
ExpandDimsactivation_2/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_5/conv1d/ExpandDims╒
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim▌
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d_5/conv1d/ExpandDims_1▄
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_5/conv1dп
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_5/conv1d/Squeeze╒
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yс
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/addж
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_3/batchnorm/Rsqrtс
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▐
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/mul╪
%batch_normalization_3/batchnorm/mul_1Mul conv1d_5/conv1d/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_3/batchnorm/mul_1█
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1▐
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_3/batchnorm/mul_2█
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2▄
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/subу
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_3/batchnorm/add_1В
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim╕
max_pooling1d_1/ExpandDims
ExpandDimsadd/add:z:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
max_pooling1d_1/ExpandDims╨
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*1
_output_shapes
:         АА*
ksize
*
paddingSAME*
strides
2
max_pooling1d_1/MaxPoolо
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims
2
max_pooling1d_1/SqueezeС
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2
activation_3/ReluЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_6/conv1d/ExpandDims/dim╠
conv1d_6/conv1d/ExpandDims
ExpandDimsactivation_3/Relu:activations:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d_6/conv1d/ExpandDims╒
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:──*
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim▌
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:──2
conv1d_6/conv1d/ExpandDims_1▄
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_6/conv1dп
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_6/conv1d/SqueezeЛ
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_4/conv1d/ExpandDims/dim═
conv1d_4/conv1d/ExpandDims
ExpandDims max_pooling1d_1/Squeeze:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_4/conv1d/ExpandDims╒
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim▌
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d_4/conv1d/ExpandDims_1▄
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_4/conv1dп
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_4/conv1d/SqueezeЫ
	add_1/addAddV2 conv1d_6/conv1d/Squeeze:output:0 conv1d_4/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2
	add_1/add╒
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yс
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_4/batchnorm/Rsqrtс
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp▐
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/mul┼
%batch_normalization_4/batchnorm/mul_1Muladd_1/add:z:0'batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_4/batchnorm/mul_1█
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1▐
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_4/batchnorm/mul_2█
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2▄
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/subу
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_4/batchnorm/add_1С
activation_4/ReluRelu)batch_normalization_4/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2
activation_4/ReluЛ
conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_8/conv1d/ExpandDims/dim╠
conv1d_8/conv1d/ExpandDims
ExpandDimsactivation_4/Relu:activations:0'conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d_8/conv1d/ExpandDims╒
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02-
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_8/conv1d/ExpandDims_1/dim▌
conv1d_8/conv1d/ExpandDims_1
ExpandDims3conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d_8/conv1d/ExpandDims_1▄
conv1d_8/conv1dConv2D#conv1d_8/conv1d/ExpandDims:output:0%conv1d_8/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_8/conv1dп
conv1d_8/conv1d/SqueezeSqueezeconv1d_8/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_8/conv1d/Squeeze╒
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/yс
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/addж
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_5/batchnorm/Rsqrtс
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▐
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/mul╪
%batch_normalization_5/batchnorm/mul_1Mul conv1d_8/conv1d/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_5/batchnorm/mul_1█
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1▐
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_5/batchnorm/mul_2█
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2▄
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/subу
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_5/batchnorm/add_1В
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim║
max_pooling1d_2/ExpandDims
ExpandDimsadd_1/add:z:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
max_pooling1d_2/ExpandDims╧
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*0
_output_shapes
:         @─*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPoolн
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*,
_output_shapes
:         @─*
squeeze_dims
2
max_pooling1d_2/SqueezeС
activation_5/ReluRelu)batch_normalization_5/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2
activation_5/ReluЛ
conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_9/conv1d/ExpandDims/dim╠
conv1d_9/conv1d/ExpandDims
ExpandDimsactivation_5/Relu:activations:0'conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_9/conv1d/ExpandDims╒
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_9_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02-
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_9/conv1d/ExpandDims_1/dim▌
conv1d_9/conv1d/ExpandDims_1
ExpandDims3conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d_9/conv1d/ExpandDims_1█
conv1d_9/conv1dConv2D#conv1d_9/conv1d/ExpandDims:output:0%conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1d_9/conv1dо
conv1d_9/conv1d/SqueezeSqueezeconv1d_9/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d_9/conv1d/SqueezeЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_7/conv1d/ExpandDims/dim╠
conv1d_7/conv1d/ExpandDims
ExpandDims max_pooling1d_2/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @─2
conv1d_7/conv1d/ExpandDims╒
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim▌
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d_7/conv1d/ExpandDims_1█
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1d_7/conv1dо
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d_7/conv1d/SqueezeЪ
	add_2/addAddV2 conv1d_9/conv1d/Squeeze:output:0 conv1d_7/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2
	add_2/add╒
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yс
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/addж
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtс
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▐
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mul─
%batch_normalization_6/batchnorm/mul_1Muladd_2/add:z:0'batch_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2'
%batch_normalization_6/batchnorm/mul_1█
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1▐
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2█
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2▄
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/subт
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2'
%batch_normalization_6/batchnorm/add_1Р
activation_6/ReluRelu)batch_normalization_6/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @А2
activation_6/ReluН
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_11/conv1d/ExpandDims/dim╬
conv1d_11/conv1d/ExpandDims
ExpandDimsactivation_6/Relu:activations:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
conv1d_11/conv1d/ExpandDims╪
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimс
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d_11/conv1d/ExpandDims_1▀
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @└*
paddingSAME*
strides
2
conv1d_11/conv1d▒
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*,
_output_shapes
:         @└*
squeeze_dims

¤        2
conv1d_11/conv1d/Squeeze╒
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/yс
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/addж
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_7/batchnorm/Rsqrtс
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▐
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/mul╪
%batch_normalization_7/batchnorm/mul_1Mul!conv1d_11/conv1d/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2'
%batch_normalization_7/batchnorm/mul_1█
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1▐
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_7/batchnorm/mul_2█
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2▄
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/subт
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2'
%batch_normalization_7/batchnorm/add_1В
max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_3/ExpandDims/dim╣
max_pooling1d_3/ExpandDims
ExpandDimsadd_2/add:z:0'max_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
max_pooling1d_3/ExpandDims╧
max_pooling1d_3/MaxPoolMaxPool#max_pooling1d_3/ExpandDims:output:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling1d_3/MaxPoolн
max_pooling1d_3/SqueezeSqueeze max_pooling1d_3/MaxPool:output:0*
T0*,
_output_shapes
:         А*
squeeze_dims
2
max_pooling1d_3/SqueezeР
activation_7/ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @└2
activation_7/ReluН
conv1d_12/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_12/conv1d/ExpandDims/dim╬
conv1d_12/conv1d/ExpandDims
ExpandDimsactivation_7/Relu:activations:0(conv1d_12/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @└2
conv1d_12/conv1d/ExpandDims╪
,conv1d_12/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_12_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:└└*
dtype02.
,conv1d_12/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_12/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_12/conv1d/ExpandDims_1/dimс
conv1d_12/conv1d/ExpandDims_1
ExpandDims4conv1d_12/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_12/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:└└2
conv1d_12/conv1d/ExpandDims_1▀
conv1d_12/conv1dConv2D$conv1d_12/conv1d/ExpandDims:output:0&conv1d_12/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1d_12/conv1d▒
conv1d_12/conv1d/SqueezeSqueezeconv1d_12/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d_12/conv1d/SqueezeН
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_10/conv1d/ExpandDims/dim╧
conv1d_10/conv1d/ExpandDims
ExpandDims max_pooling1d_3/Squeeze:output:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d_10/conv1d/ExpandDims╪
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimс
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d_10/conv1d/ExpandDims_1▀
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1d_10/conv1d▒
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d_10/conv1d/SqueezeЬ
	add_3/addAddV2!conv1d_12/conv1d/Squeeze:output:0!conv1d_10/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2
	add_3/add╒
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOpУ
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_8/batchnorm/add/yс
#batch_normalization_8/batchnorm/addAddV26batch_normalization_8/batchnorm/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/addж
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_8/batchnorm/Rsqrtс
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp▐
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/mul─
%batch_normalization_8/batchnorm/mul_1Muladd_3/add:z:0'batch_normalization_8/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └2'
%batch_normalization_8/batchnorm/mul_1█
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_1▐
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_8/batchnorm/mul_2█
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_2▄
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/subт
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2'
%batch_normalization_8/batchnorm/add_1Р
activation_8/ReluRelu)batch_normalization_8/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └2
activation_8/Relu~
embed/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
embed/Mean/reduction_indicesЫ

embed/MeanMeanactivation_8/Relu:activations:0%embed/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         └2

embed/Meanh
IdentityIdentityembed/Mean:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А ::::::::::::::::::::::::::::::::::::::::::::::::::T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
▒╡
╚
F__inference_functional_1_layer_call_and_return_conditional_losses_4112
ecg
conv1d_2808
batch_normalization_2893
batch_normalization_2895
batch_normalization_2897
batch_normalization_2899
conv1d_2_2936
batch_normalization_1_3021
batch_normalization_1_3023
batch_normalization_1_3025
batch_normalization_1_3027
conv1d_3_3065
conv1d_1_3089
batch_normalization_2_3189
batch_normalization_2_3191
batch_normalization_2_3193
batch_normalization_2_3195
conv1d_5_3232
batch_normalization_3_3317
batch_normalization_3_3319
batch_normalization_3_3321
batch_normalization_3_3323
conv1d_6_3361
conv1d_4_3385
batch_normalization_4_3485
batch_normalization_4_3487
batch_normalization_4_3489
batch_normalization_4_3491
conv1d_8_3528
batch_normalization_5_3613
batch_normalization_5_3615
batch_normalization_5_3617
batch_normalization_5_3619
conv1d_9_3657
conv1d_7_3681
batch_normalization_6_3781
batch_normalization_6_3783
batch_normalization_6_3785
batch_normalization_6_3787
conv1d_11_3824
batch_normalization_7_3909
batch_normalization_7_3911
batch_normalization_7_3913
batch_normalization_7_3915
conv1d_12_3953
conv1d_10_3977
batch_normalization_8_4077
batch_normalization_8_4079
batch_normalization_8_4081
batch_normalization_8_4083
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв!conv1d_10/StatefulPartitionedCallв!conv1d_11/StatefulPartitionedCallв!conv1d_12/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв conv1d_8/StatefulPartitionedCallв conv1d_9/StatefulPartitionedCallў
conv1d/StatefulPartitionedCallStatefulPartitionedCallecgconv1d_2808*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_27992 
conv1d/StatefulPartitionedCallб
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_2893batch_normalization_2895batch_normalization_2897batch_normalization_2899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28462-
+batch_normalization/StatefulPartitionedCallЛ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_29072
activation/PartitionedCallа
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv1d_2_2936*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_29272"
 conv1d_2/StatefulPartitionedCall▓
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_1_3021batch_normalization_1_3023batch_normalization_1_3025batch_normalization_1_3027*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29742/
-batch_normalization_1/StatefulPartitionedCallГ
max_pooling1d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_17332
max_pooling1d/PartitionedCallФ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_30362
activation_1/PartitionedCallв
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv1d_3_3065*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_30562"
 conv1d_3/StatefulPartitionedCallг
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_3089*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_30802"
 conv1d_1/StatefulPartitionedCallШ
add/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_30982
add/PartitionedCallе
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_2_3189batch_normalization_2_3191batch_normalization_2_3193batch_normalization_2_3195*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31422/
-batch_normalization_2/StatefulPartitionedCallФ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_32032
activation_2/PartitionedCallв
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv1d_5_3232*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_5_layer_call_and_return_conditional_losses_32232"
 conv1d_5/StatefulPartitionedCall▓
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_3317batch_normalization_3_3319batch_normalization_3_3321batch_normalization_3_3323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32702/
-batch_normalization_3/StatefulPartitionedCallГ
max_pooling1d_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_20282!
max_pooling1d_1/PartitionedCallФ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_33322
activation_3/PartitionedCallв
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv1d_6_3361*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_6_layer_call_and_return_conditional_losses_33522"
 conv1d_6/StatefulPartitionedCallе
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_4_3385*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_4_layer_call_and_return_conditional_losses_33762"
 conv1d_4/StatefulPartitionedCallЮ
add_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33942
add_1/PartitionedCallз
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_4_3485batch_normalization_4_3487batch_normalization_4_3489batch_normalization_4_3491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34382/
-batch_normalization_4/StatefulPartitionedCallФ
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_34992
activation_4/PartitionedCallв
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv1d_8_3528*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_8_layer_call_and_return_conditional_losses_35192"
 conv1d_8/StatefulPartitionedCall▓
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0batch_normalization_5_3613batch_normalization_5_3615batch_normalization_5_3617batch_normalization_5_3619*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35662/
-batch_normalization_5/StatefulPartitionedCallД
max_pooling1d_2/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23232!
max_pooling1d_2/PartitionedCallФ
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_36282
activation_5/PartitionedCallб
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv1d_9_3657*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_9_layer_call_and_return_conditional_losses_36482"
 conv1d_9/StatefulPartitionedCallд
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_7_3681*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_7_layer_call_and_return_conditional_losses_36722"
 conv1d_7/StatefulPartitionedCallЭ
add_2/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_2_layer_call_and_return_conditional_losses_36902
add_2/PartitionedCallж
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_6_3781batch_normalization_6_3783batch_normalization_6_3785batch_normalization_6_3787*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37342/
-batch_normalization_6/StatefulPartitionedCallУ
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_37952
activation_6/PartitionedCallе
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv1d_11_3824*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_11_layer_call_and_return_conditional_losses_38152#
!conv1d_11/StatefulPartitionedCall▓
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_7_3909batch_normalization_7_3911batch_normalization_7_3913batch_normalization_7_3915*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38622/
-batch_normalization_7/StatefulPartitionedCallД
max_pooling1d_3/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_26182!
max_pooling1d_3/PartitionedCallУ
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_39242
activation_7/PartitionedCallе
!conv1d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv1d_12_3953*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_12_layer_call_and_return_conditional_losses_39442#
!conv1d_12/StatefulPartitionedCallи
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_3/PartitionedCall:output:0conv1d_10_3977*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_10_layer_call_and_return_conditional_losses_39682#
!conv1d_10/StatefulPartitionedCallЯ
add_3/PartitionedCallPartitionedCall*conv1d_12/StatefulPartitionedCall:output:0*conv1d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_3_layer_call_and_return_conditional_losses_39862
add_3/PartitionedCallж
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_8_4077batch_normalization_8_4079batch_normalization_8_4081batch_normalization_8_4083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40302/
-batch_normalization_8/StatefulPartitionedCallУ
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_40912
activation_8/PartitionedCallщ
embed/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_41042
embed/PartitionedCallщ
IdentityIdentityembed/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall"^conv1d_12/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2F
!conv1d_12/StatefulPartitionedCall!conv1d_12/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall:Q M
,
_output_shapes
:         А 

_user_specified_nameecg
и
Т
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2994

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А:::::U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╚
b
F__inference_activation_6_layer_call_and_return_conditional_losses_7205

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         @А2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @А:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2720

inputs
assignmovingavg_2695
assignmovingavg_1_2701)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2695*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2695*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2695*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2695*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2695AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2695*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2701*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2701*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2701*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2701*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2701AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2701*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╗
i
?__inference_add_3_layer_call_and_return_conditional_losses_3986

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:         └2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         └:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs:TP
,
_output_shapes
:         └
 
_user_specified_nameinputs
о
G
+__inference_activation_5_layer_call_fn_6986

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_36282
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
у
c
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_1733

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Д
Т
B__inference_conv1d_9_layer_call_and_return_conditional_losses_6998

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3862

inputs
assignmovingavg_3837
assignmovingavg_1_3843)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         @└2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3837*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3837*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3837*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3837*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3837AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3837*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3843*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3843*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3843*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3843*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3843AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3843*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4030

inputs
assignmovingavg_4005
assignmovingavg_1_4011)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/4005*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4005*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/4005*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/4005*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4005AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/4005*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/4011*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4011*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/4011*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/4011*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4011AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/4011*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
у
█
"__inference_signature_wrapper_4850
ecg
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallecgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./01*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_14442
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
,
_output_shapes
:         А 

_user_specified_nameecg
щ
з
4__inference_batch_normalization_8_layer_call_fn_7604

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27202
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╦
з
4__inference_batch_normalization_3_layer_call_fn_6559

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32902
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2974

inputs
assignmovingavg_2949
assignmovingavg_1_2955)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2949*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2949*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2949*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2949*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2949AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2949*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2955*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2955*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2955*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2955*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2955AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2955*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
В
У
C__inference_conv1d_11_layer_call_and_return_conditional_losses_3815

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @└*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @└*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @А::T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
В
Т
B__inference_conv1d_2_layer_call_and_return_conditional_losses_5971

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╕
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А А*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А А*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А @::T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2565

inputs
assignmovingavg_2540
assignmovingavg_1_2546)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2540*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2540*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2540*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2540*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2540AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2540*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2546*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2546*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2546*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2546*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2546AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2546*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
ї
J
.__inference_max_pooling1d_1_layer_call_fn_2034

inputs
identity▌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_20282
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
√
Р
@__inference_conv1d_layer_call_and_return_conditional_losses_5778

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А 2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         А @*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         А @*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А ::T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
к
G
+__inference_activation_7_layer_call_fn_7403

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_39242
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @└:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╬
n
(__inference_conv1d_11_layer_call_fn_7229

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_11_layer_call_and_return_conditional_losses_38152
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @А:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
╔
з
4__inference_batch_normalization_4_layer_call_fn_6770

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34382
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╞
`
D__inference_activation_layer_call_and_return_conditional_losses_2907

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         А @2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А @:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1713

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7591

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └:::::] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╠
b
F__inference_activation_1_layer_call_and_return_conditional_losses_3036

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А А2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А А:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╚
b
F__inference_activation_7_layer_call_and_return_conditional_losses_3924

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         @└2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @└:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
║)
┬
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5821

inputs
assignmovingavg_5796
assignmovingavg_1_5802)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         А @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/5796*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5796*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/5796*
_output_shapes
:@2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/5796*
_output_shapes
:@2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5796AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/5796*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/5802*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5802*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5802*
_output_shapes
:@2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5802*
_output_shapes
:@2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5802AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/5802*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1975

inputs
assignmovingavg_1950
assignmovingavg_1_1956)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/1950*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1950*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/1950*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/1950*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1950AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/1950*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/1956*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1956*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1956*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1956*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1956AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/1956*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2130

inputs
assignmovingavg_2105
assignmovingavg_1_2111)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2105*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2105*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2105*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2105*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2105AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2105*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2111*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2111*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2111*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2111*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2111AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2111*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
к
G
+__inference_activation_8_layer_call_fn_7627

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_40912
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_7_layer_call_fn_7393

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  └*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_25982
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╟
з
4__inference_batch_normalization_7_layer_call_fn_7311

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38822
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6034

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А:::::U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6096

inputs
assignmovingavg_6071
assignmovingavg_1_6077)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6071*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6071*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6071*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6071*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6071AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6071*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6077*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6077*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6077*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6077*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6077AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6077*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
щ
з
4__inference_batch_normalization_4_layer_call_fn_6688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_21302
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
╔
з
4__inference_batch_normalization_2_layer_call_fn_6271

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31422
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
Е╤
ю
 __inference__traced_restore_7976
file_prefix"
assignvariableop_conv1d_kernel0
,assignvariableop_1_batch_normalization_gamma/
+assignvariableop_2_batch_normalization_beta6
2assignvariableop_3_batch_normalization_moving_mean:
6assignvariableop_4_batch_normalization_moving_variance&
"assignvariableop_5_conv1d_2_kernel2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta8
4assignvariableop_8_batch_normalization_1_moving_mean<
8assignvariableop_9_batch_normalization_1_moving_variance'
#assignvariableop_10_conv1d_3_kernel'
#assignvariableop_11_conv1d_1_kernel3
/assignvariableop_12_batch_normalization_2_gamma2
.assignvariableop_13_batch_normalization_2_beta9
5assignvariableop_14_batch_normalization_2_moving_mean=
9assignvariableop_15_batch_normalization_2_moving_variance'
#assignvariableop_16_conv1d_5_kernel3
/assignvariableop_17_batch_normalization_3_gamma2
.assignvariableop_18_batch_normalization_3_beta9
5assignvariableop_19_batch_normalization_3_moving_mean=
9assignvariableop_20_batch_normalization_3_moving_variance'
#assignvariableop_21_conv1d_6_kernel'
#assignvariableop_22_conv1d_4_kernel3
/assignvariableop_23_batch_normalization_4_gamma2
.assignvariableop_24_batch_normalization_4_beta9
5assignvariableop_25_batch_normalization_4_moving_mean=
9assignvariableop_26_batch_normalization_4_moving_variance'
#assignvariableop_27_conv1d_8_kernel3
/assignvariableop_28_batch_normalization_5_gamma2
.assignvariableop_29_batch_normalization_5_beta9
5assignvariableop_30_batch_normalization_5_moving_mean=
9assignvariableop_31_batch_normalization_5_moving_variance'
#assignvariableop_32_conv1d_9_kernel'
#assignvariableop_33_conv1d_7_kernel3
/assignvariableop_34_batch_normalization_6_gamma2
.assignvariableop_35_batch_normalization_6_beta9
5assignvariableop_36_batch_normalization_6_moving_mean=
9assignvariableop_37_batch_normalization_6_moving_variance(
$assignvariableop_38_conv1d_11_kernel3
/assignvariableop_39_batch_normalization_7_gamma2
.assignvariableop_40_batch_normalization_7_beta9
5assignvariableop_41_batch_normalization_7_moving_mean=
9assignvariableop_42_batch_normalization_7_moving_variance(
$assignvariableop_43_conv1d_12_kernel(
$assignvariableop_44_conv1d_10_kernel3
/assignvariableop_45_batch_normalization_8_gamma2
.assignvariableop_46_batch_normalization_8_beta9
5assignvariableop_47_batch_normalization_8_moving_mean=
9assignvariableop_48_batch_normalization_8_moving_variance
identity_50ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9▌
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*щ
value▀B▄2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesи
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▐
_output_shapes╦
╚::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
4222
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▒
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2░
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╖
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╗
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5з
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv1d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6│
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▓
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╜
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10л
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11л
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv1d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╖
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╢
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╜
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┴
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16л
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv1d_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╖
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╢
AssignVariableOp_18AssignVariableOp.assignvariableop_18_batch_normalization_3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╜
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_3_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┴
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_3_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21л
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv1d_6_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22л
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv1d_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╖
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_4_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╢
AssignVariableOp_24AssignVariableOp.assignvariableop_24_batch_normalization_4_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╜
AssignVariableOp_25AssignVariableOp5assignvariableop_25_batch_normalization_4_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┴
AssignVariableOp_26AssignVariableOp9assignvariableop_26_batch_normalization_4_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27л
AssignVariableOp_27AssignVariableOp#assignvariableop_27_conv1d_8_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╖
AssignVariableOp_28AssignVariableOp/assignvariableop_28_batch_normalization_5_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╢
AssignVariableOp_29AssignVariableOp.assignvariableop_29_batch_normalization_5_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╜
AssignVariableOp_30AssignVariableOp5assignvariableop_30_batch_normalization_5_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31┴
AssignVariableOp_31AssignVariableOp9assignvariableop_31_batch_normalization_5_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32л
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv1d_9_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33л
AssignVariableOp_33AssignVariableOp#assignvariableop_33_conv1d_7_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╖
AssignVariableOp_34AssignVariableOp/assignvariableop_34_batch_normalization_6_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╢
AssignVariableOp_35AssignVariableOp.assignvariableop_35_batch_normalization_6_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36╜
AssignVariableOp_36AssignVariableOp5assignvariableop_36_batch_normalization_6_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37┴
AssignVariableOp_37AssignVariableOp9assignvariableop_37_batch_normalization_6_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38м
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv1d_11_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╖
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_7_gammaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╢
AssignVariableOp_40AssignVariableOp.assignvariableop_40_batch_normalization_7_betaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╜
AssignVariableOp_41AssignVariableOp5assignvariableop_41_batch_normalization_7_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42┴
AssignVariableOp_42AssignVariableOp9assignvariableop_42_batch_normalization_7_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43м
AssignVariableOp_43AssignVariableOp$assignvariableop_43_conv1d_12_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44м
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv1d_10_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╖
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_8_gammaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╢
AssignVariableOp_46AssignVariableOp.assignvariableop_46_batch_normalization_8_betaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╜
AssignVariableOp_47AssignVariableOp5assignvariableop_47_batch_normalization_8_moving_meanIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48┴
AssignVariableOp_48AssignVariableOp9assignvariableop_48_batch_normalization_8_moving_varianceIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpФ	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49З	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*█
_input_shapes╔
╞: :::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╤
Т
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6675

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─:::::] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_8_layer_call_and_return_conditional_losses_6805

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
┼
з
4__inference_batch_normalization_7_layer_call_fn_7298

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38622
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3438

inputs
assignmovingavg_3413
assignmovingavg_1_3419)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3413*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3413*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3413*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3413*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3413AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3413*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3419*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3419*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3419*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3419*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3419AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3419*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3566

inputs
assignmovingavg_3541
assignmovingavg_1_3547)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         АА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3541*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3541*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3541*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3541*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3541AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3541*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3547*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3547*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3547*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3547*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3547AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3547*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
о
G
+__inference_activation_3_layer_call_fn_6569

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_33322
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
х
e
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2618

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6116

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7367

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └:::::] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
В
Т
B__inference_conv1d_2_layer_call_and_return_conditional_losses_2927

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╕
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А А*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А А*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А @::T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
╔
з
4__inference_batch_normalization_3_layer_call_fn_6546

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32702
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2458

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
у
е
2__inference_batch_normalization_layer_call_fn_5949

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_15732
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_5_layer_call_fn_6976

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23032
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4050

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └:::::T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6513

inputs
assignmovingavg_6488
assignmovingavg_1_6494)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6488*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6488*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6488*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6488*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6488AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6488*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6494*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6494*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6494*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6494*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6494AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6494*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
Ь
ч
+__inference_functional_1_layer_call_fn_5766

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./01*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_46442
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_6_layer_call_and_return_conditional_losses_3352

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:──*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:──2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_4_layer_call_and_return_conditional_losses_6600

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╬
m
'__inference_conv1d_2_layer_call_fn_5978

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_29272
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А @:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6320

inputs
assignmovingavg_6295
assignmovingavg_1_6301)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6295*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6295*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6295*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6295*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6295AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6295*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6301*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6301*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6301*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6301*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6301AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6301*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
√
[
?__inference_embed_layer_call_and_return_conditional_losses_7644

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Ш
Р
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5841

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @:::::T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
┼
з
4__inference_batch_normalization_8_layer_call_fn_7522

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40302
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
┴
i
?__inference_add_1_layer_call_and_return_conditional_losses_3394

inputs
inputs_1
identity]
addAddV2inputsinputs_1*
T0*-
_output_shapes
:         А─2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         А─:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3270

inputs
assignmovingavg_3245
assignmovingavg_1_3251)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3245*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3245*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3245*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3245*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3245AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3245*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3251*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3251*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3251*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3251*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3251AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3251*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_5_layer_call_and_return_conditional_losses_6388

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╬
n
(__inference_conv1d_12_layer_call_fn_7422

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_12_layer_call_and_return_conditional_losses_39442
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @└:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7509

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └:::::T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_6_layer_call_fn_7200

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_24582
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3734

inputs
assignmovingavg_3709
assignmovingavg_1_3715)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         @А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3709*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3709*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3709*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3709*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3709AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3709*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3715*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3715*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3715*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3715*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3715AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3715*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2753

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └:::::] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7265

inputs
assignmovingavg_7240
assignmovingavg_1_7246)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         @└2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7240*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7240*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7240*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7240*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7240AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7240*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7246*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7246*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7246*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7246*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7246AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7246*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6930

inputs
assignmovingavg_6905
assignmovingavg_1_6911)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6905*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6905*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6905*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6905*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6905AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6905*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6911*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6911*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6911*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6911*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6911AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6911*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7347

inputs
assignmovingavg_7322
assignmovingavg_1_7328)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7322*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7322*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7322*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7322*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7322AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7322*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7328*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7328*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7328*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7328*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7328AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7328*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╠
b
F__inference_activation_4_layer_call_and_return_conditional_losses_6788

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А─2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
├
k
?__inference_add_2_layer_call_and_return_conditional_losses_7030
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:         @А2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         @А:         @А:V R
,
_output_shapes
:         @А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         @А
"
_user_specified_name
inputs/1
╤
Т
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2163

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─:::::] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6868

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА:::::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_3_layer_call_fn_6477

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_20082
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6451

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─:::::] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2598

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └:::::] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
н
N
"__inference_add_layer_call_fn_6202
inputs_0
inputs_1
identity╬
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_30982
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         АА:         АА:W S
-
_output_shapes
:         АА
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         АА
"
_user_specified_name
inputs/1
╦
з
4__inference_batch_normalization_2_layer_call_fn_6284

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31622
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1835

inputs
assignmovingavg_1810
assignmovingavg_1_1816)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/1810*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1810*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/1810*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/1810*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1810AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/1810*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/1816*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1816*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1816*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1816*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1816AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/1816*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
║)
┬
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2846

inputs
assignmovingavg_2821
assignmovingavg_1_2827)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         А @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2821*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2821*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2821*
_output_shapes
:@2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2821*
_output_shapes
:@2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2821AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2821*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2827*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2827*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2827*
_output_shapes
:@2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2827*
_output_shapes
:@2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2827AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2827*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
ж
E
)__inference_activation_layer_call_fn_5959

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_29072
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А @:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
л
P
$__inference_add_3_layer_call_fn_7453
inputs_0
inputs_1
identity╧
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_3_layer_call_and_return_conditional_losses_39862
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         └:         └:V R
,
_output_shapes
:         └
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         └
"
_user_specified_name
inputs/1
к
G
+__inference_activation_6_layer_call_fn_7210

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_37952
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @А:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3586

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА:::::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3142

inputs
assignmovingavg_3117
assignmovingavg_1_3123)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         АА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/3117*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3117*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/3117*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/3117*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3117AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/3117*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/3123*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3123*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3123*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/3123*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3123AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/3123*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2270

inputs
assignmovingavg_2245
assignmovingavg_1_2251)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2245*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2245*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2245*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2245*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2245AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2245*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2251*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2251*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2251*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2251*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2251AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2251*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3458

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─:::::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╠
b
F__inference_activation_4_layer_call_and_return_conditional_losses_3499

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А─2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╔
з
4__inference_batch_normalization_5_layer_call_fn_6881

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35662
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╟
з
4__inference_batch_normalization_8_layer_call_fn_7535

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40502
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
о
G
+__inference_activation_1_layer_call_fn_6152

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_30362
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А А:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
В
У
C__inference_conv1d_12_layer_call_and_return_conditional_losses_7415

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @└2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:└└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:└└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @└::T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
У
ф
+__inference_functional_1_layer_call_fn_4745
ecg
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallecgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./01*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_46442
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
,
_output_shapes
:         А 

_user_specified_nameecg
ы
з
4__inference_batch_normalization_4_layer_call_fn_6701

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_21632
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
В
У
C__inference_conv1d_11_layer_call_and_return_conditional_losses_7222

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @└*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @└*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @А::T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6340

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╞
`
D__inference_activation_layer_call_and_return_conditional_losses_5954

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         А @2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А @:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
╚
b
F__inference_activation_8_layer_call_and_return_conditional_losses_4091

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         └2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
Д
Т
B__inference_conv1d_9_layer_call_and_return_conditional_losses_3648

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
┴
е
2__inference_batch_normalization_layer_call_fn_5854

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28462
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
╠╡
╦
F__inference_functional_1_layer_call_and_return_conditional_losses_4644

inputs
conv1d_4505
batch_normalization_4508
batch_normalization_4510
batch_normalization_4512
batch_normalization_4514
conv1d_2_4518
batch_normalization_1_4521
batch_normalization_1_4523
batch_normalization_1_4525
batch_normalization_1_4527
conv1d_3_4532
conv1d_1_4535
batch_normalization_2_4539
batch_normalization_2_4541
batch_normalization_2_4543
batch_normalization_2_4545
conv1d_5_4549
batch_normalization_3_4552
batch_normalization_3_4554
batch_normalization_3_4556
batch_normalization_3_4558
conv1d_6_4563
conv1d_4_4566
batch_normalization_4_4570
batch_normalization_4_4572
batch_normalization_4_4574
batch_normalization_4_4576
conv1d_8_4580
batch_normalization_5_4583
batch_normalization_5_4585
batch_normalization_5_4587
batch_normalization_5_4589
conv1d_9_4594
conv1d_7_4597
batch_normalization_6_4601
batch_normalization_6_4603
batch_normalization_6_4605
batch_normalization_6_4607
conv1d_11_4611
batch_normalization_7_4614
batch_normalization_7_4616
batch_normalization_7_4618
batch_normalization_7_4620
conv1d_12_4625
conv1d_10_4628
batch_normalization_8_4632
batch_normalization_8_4634
batch_normalization_8_4636
batch_normalization_8_4638
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв!conv1d_10/StatefulPartitionedCallв!conv1d_11/StatefulPartitionedCallв!conv1d_12/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв conv1d_8/StatefulPartitionedCallв conv1d_9/StatefulPartitionedCall·
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_4505*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_27992 
conv1d/StatefulPartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_4508batch_normalization_4510batch_normalization_4512batch_normalization_4514*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28662-
+batch_normalization/StatefulPartitionedCallЛ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_29072
activation/PartitionedCallа
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv1d_2_4518*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_29272"
 conv1d_2/StatefulPartitionedCall┤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_1_4521batch_normalization_1_4523batch_normalization_1_4525batch_normalization_1_4527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29942/
-batch_normalization_1/StatefulPartitionedCallГ
max_pooling1d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_17332
max_pooling1d/PartitionedCallФ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_30362
activation_1/PartitionedCallв
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv1d_3_4532*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_30562"
 conv1d_3/StatefulPartitionedCallг
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_4535*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_30802"
 conv1d_1/StatefulPartitionedCallШ
add/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_30982
add/PartitionedCallз
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_2_4539batch_normalization_2_4541batch_normalization_2_4543batch_normalization_2_4545*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31622/
-batch_normalization_2/StatefulPartitionedCallФ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_32032
activation_2/PartitionedCallв
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv1d_5_4549*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_5_layer_call_and_return_conditional_losses_32232"
 conv1d_5/StatefulPartitionedCall┤
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_4552batch_normalization_3_4554batch_normalization_3_4556batch_normalization_3_4558*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32902/
-batch_normalization_3/StatefulPartitionedCallГ
max_pooling1d_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_20282!
max_pooling1d_1/PartitionedCallФ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_33322
activation_3/PartitionedCallв
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv1d_6_4563*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_6_layer_call_and_return_conditional_losses_33522"
 conv1d_6/StatefulPartitionedCallе
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_4_4566*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_4_layer_call_and_return_conditional_losses_33762"
 conv1d_4/StatefulPartitionedCallЮ
add_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33942
add_1/PartitionedCallй
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_4_4570batch_normalization_4_4572batch_normalization_4_4574batch_normalization_4_4576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582/
-batch_normalization_4/StatefulPartitionedCallФ
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_34992
activation_4/PartitionedCallв
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv1d_8_4580*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_8_layer_call_and_return_conditional_losses_35192"
 conv1d_8/StatefulPartitionedCall┤
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0batch_normalization_5_4583batch_normalization_5_4585batch_normalization_5_4587batch_normalization_5_4589*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35862/
-batch_normalization_5/StatefulPartitionedCallД
max_pooling1d_2/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23232!
max_pooling1d_2/PartitionedCallФ
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_36282
activation_5/PartitionedCallб
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv1d_9_4594*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_9_layer_call_and_return_conditional_losses_36482"
 conv1d_9/StatefulPartitionedCallд
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_7_4597*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_7_layer_call_and_return_conditional_losses_36722"
 conv1d_7/StatefulPartitionedCallЭ
add_2/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_2_layer_call_and_return_conditional_losses_36902
add_2/PartitionedCallи
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_6_4601batch_normalization_6_4603batch_normalization_6_4605batch_normalization_6_4607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37542/
-batch_normalization_6/StatefulPartitionedCallУ
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_37952
activation_6/PartitionedCallе
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv1d_11_4611*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_11_layer_call_and_return_conditional_losses_38152#
!conv1d_11/StatefulPartitionedCall┤
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_7_4614batch_normalization_7_4616batch_normalization_7_4618batch_normalization_7_4620*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38822/
-batch_normalization_7/StatefulPartitionedCallД
max_pooling1d_3/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_26182!
max_pooling1d_3/PartitionedCallУ
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_39242
activation_7/PartitionedCallе
!conv1d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv1d_12_4625*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_12_layer_call_and_return_conditional_losses_39442#
!conv1d_12/StatefulPartitionedCallи
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_3/PartitionedCall:output:0conv1d_10_4628*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_10_layer_call_and_return_conditional_losses_39682#
!conv1d_10/StatefulPartitionedCallЯ
add_3/PartitionedCallPartitionedCall*conv1d_12/StatefulPartitionedCall:output:0*conv1d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_3_layer_call_and_return_conditional_losses_39862
add_3/PartitionedCallи
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_8_4632batch_normalization_8_4634batch_normalization_8_4636batch_normalization_8_4638*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40502/
-batch_normalization_8/StatefulPartitionedCallУ
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_40912
activation_8/PartitionedCallщ
embed/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_41042
embed/PartitionedCallщ
IdentityIdentityembed/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall"^conv1d_12/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2F
!conv1d_12/StatefulPartitionedCall!conv1d_12/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6655

inputs
assignmovingavg_6630
assignmovingavg_1_6636)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6630*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6630AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6636*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6636AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
╠
b
F__inference_activation_5_layer_call_and_return_conditional_losses_3628

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         АА2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╠
b
F__inference_activation_3_layer_call_and_return_conditional_losses_3332

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А─2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╦
з
4__inference_batch_normalization_5_layer_call_fn_6894

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35862
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╚
b
F__inference_activation_6_layer_call_and_return_conditional_losses_3795

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         @А2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @А:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
├╡
╚
F__inference_functional_1_layer_call_and_return_conditional_losses_4254
ecg
conv1d_4115
batch_normalization_4118
batch_normalization_4120
batch_normalization_4122
batch_normalization_4124
conv1d_2_4128
batch_normalization_1_4131
batch_normalization_1_4133
batch_normalization_1_4135
batch_normalization_1_4137
conv1d_3_4142
conv1d_1_4145
batch_normalization_2_4149
batch_normalization_2_4151
batch_normalization_2_4153
batch_normalization_2_4155
conv1d_5_4159
batch_normalization_3_4162
batch_normalization_3_4164
batch_normalization_3_4166
batch_normalization_3_4168
conv1d_6_4173
conv1d_4_4176
batch_normalization_4_4180
batch_normalization_4_4182
batch_normalization_4_4184
batch_normalization_4_4186
conv1d_8_4190
batch_normalization_5_4193
batch_normalization_5_4195
batch_normalization_5_4197
batch_normalization_5_4199
conv1d_9_4204
conv1d_7_4207
batch_normalization_6_4211
batch_normalization_6_4213
batch_normalization_6_4215
batch_normalization_6_4217
conv1d_11_4221
batch_normalization_7_4224
batch_normalization_7_4226
batch_normalization_7_4228
batch_normalization_7_4230
conv1d_12_4235
conv1d_10_4238
batch_normalization_8_4242
batch_normalization_8_4244
batch_normalization_8_4246
batch_normalization_8_4248
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв!conv1d_10/StatefulPartitionedCallв!conv1d_11/StatefulPartitionedCallв!conv1d_12/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв conv1d_8/StatefulPartitionedCallв conv1d_9/StatefulPartitionedCallў
conv1d/StatefulPartitionedCallStatefulPartitionedCallecgconv1d_4115*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_27992 
conv1d/StatefulPartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_4118batch_normalization_4120batch_normalization_4122batch_normalization_4124*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28662-
+batch_normalization/StatefulPartitionedCallЛ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_29072
activation/PartitionedCallа
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv1d_2_4128*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_29272"
 conv1d_2/StatefulPartitionedCall┤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_1_4131batch_normalization_1_4133batch_normalization_1_4135batch_normalization_1_4137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29942/
-batch_normalization_1/StatefulPartitionedCallГ
max_pooling1d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_17332
max_pooling1d/PartitionedCallФ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_30362
activation_1/PartitionedCallв
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv1d_3_4142*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_30562"
 conv1d_3/StatefulPartitionedCallг
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_4145*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_30802"
 conv1d_1/StatefulPartitionedCallШ
add/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_30982
add/PartitionedCallз
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_2_4149batch_normalization_2_4151batch_normalization_2_4153batch_normalization_2_4155*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31622/
-batch_normalization_2/StatefulPartitionedCallФ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_32032
activation_2/PartitionedCallв
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv1d_5_4159*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_5_layer_call_and_return_conditional_losses_32232"
 conv1d_5/StatefulPartitionedCall┤
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_4162batch_normalization_3_4164batch_normalization_3_4166batch_normalization_3_4168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32902/
-batch_normalization_3/StatefulPartitionedCallГ
max_pooling1d_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_20282!
max_pooling1d_1/PartitionedCallФ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_33322
activation_3/PartitionedCallв
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv1d_6_4173*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_6_layer_call_and_return_conditional_losses_33522"
 conv1d_6/StatefulPartitionedCallе
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_4_4176*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_4_layer_call_and_return_conditional_losses_33762"
 conv1d_4/StatefulPartitionedCallЮ
add_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33942
add_1/PartitionedCallй
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_4_4180batch_normalization_4_4182batch_normalization_4_4184batch_normalization_4_4186*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582/
-batch_normalization_4/StatefulPartitionedCallФ
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_34992
activation_4/PartitionedCallв
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv1d_8_4190*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_8_layer_call_and_return_conditional_losses_35192"
 conv1d_8/StatefulPartitionedCall┤
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0batch_normalization_5_4193batch_normalization_5_4195batch_normalization_5_4197batch_normalization_5_4199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35862/
-batch_normalization_5/StatefulPartitionedCallД
max_pooling1d_2/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23232!
max_pooling1d_2/PartitionedCallФ
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_36282
activation_5/PartitionedCallб
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv1d_9_4204*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_9_layer_call_and_return_conditional_losses_36482"
 conv1d_9/StatefulPartitionedCallд
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_7_4207*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_7_layer_call_and_return_conditional_losses_36722"
 conv1d_7/StatefulPartitionedCallЭ
add_2/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_2_layer_call_and_return_conditional_losses_36902
add_2/PartitionedCallи
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_6_4211batch_normalization_6_4213batch_normalization_6_4215batch_normalization_6_4217*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37542/
-batch_normalization_6/StatefulPartitionedCallУ
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_37952
activation_6/PartitionedCallе
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv1d_11_4221*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_11_layer_call_and_return_conditional_losses_38152#
!conv1d_11/StatefulPartitionedCall┤
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_7_4224batch_normalization_7_4226batch_normalization_7_4228batch_normalization_7_4230*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38822/
-batch_normalization_7/StatefulPartitionedCallД
max_pooling1d_3/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_26182!
max_pooling1d_3/PartitionedCallУ
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_39242
activation_7/PartitionedCallе
!conv1d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv1d_12_4235*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_12_layer_call_and_return_conditional_losses_39442#
!conv1d_12/StatefulPartitionedCallи
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_3/PartitionedCall:output:0conv1d_10_4238*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_10_layer_call_and_return_conditional_losses_39682#
!conv1d_10/StatefulPartitionedCallЯ
add_3/PartitionedCallPartitionedCall*conv1d_12/StatefulPartitionedCall:output:0*conv1d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_3_layer_call_and_return_conditional_losses_39862
add_3/PartitionedCallи
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_8_4242batch_normalization_8_4244batch_normalization_8_4246batch_normalization_8_4248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40502/
-batch_normalization_8/StatefulPartitionedCallУ
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_40912
activation_8/PartitionedCallщ
embed/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_41042
embed/PartitionedCallщ
IdentityIdentityembed/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall"^conv1d_12/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2F
!conv1d_12/StatefulPartitionedCall!conv1d_12/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall:Q M
,
_output_shapes
:         А 

_user_specified_nameecg
ї
J
.__inference_max_pooling1d_2_layer_call_fn_2329

inputs
identity▌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23232
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_4_layer_call_and_return_conditional_losses_3376

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_5_layer_call_and_return_conditional_losses_3223

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_8_layer_call_and_return_conditional_losses_3519

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
║╡
╦
F__inference_functional_1_layer_call_and_return_conditional_losses_4399

inputs
conv1d_4260
batch_normalization_4263
batch_normalization_4265
batch_normalization_4267
batch_normalization_4269
conv1d_2_4273
batch_normalization_1_4276
batch_normalization_1_4278
batch_normalization_1_4280
batch_normalization_1_4282
conv1d_3_4287
conv1d_1_4290
batch_normalization_2_4294
batch_normalization_2_4296
batch_normalization_2_4298
batch_normalization_2_4300
conv1d_5_4304
batch_normalization_3_4307
batch_normalization_3_4309
batch_normalization_3_4311
batch_normalization_3_4313
conv1d_6_4318
conv1d_4_4321
batch_normalization_4_4325
batch_normalization_4_4327
batch_normalization_4_4329
batch_normalization_4_4331
conv1d_8_4335
batch_normalization_5_4338
batch_normalization_5_4340
batch_normalization_5_4342
batch_normalization_5_4344
conv1d_9_4349
conv1d_7_4352
batch_normalization_6_4356
batch_normalization_6_4358
batch_normalization_6_4360
batch_normalization_6_4362
conv1d_11_4366
batch_normalization_7_4369
batch_normalization_7_4371
batch_normalization_7_4373
batch_normalization_7_4375
conv1d_12_4380
conv1d_10_4383
batch_normalization_8_4387
batch_normalization_8_4389
batch_normalization_8_4391
batch_normalization_8_4393
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв!conv1d_10/StatefulPartitionedCallв!conv1d_11/StatefulPartitionedCallв!conv1d_12/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallв conv1d_3/StatefulPartitionedCallв conv1d_4/StatefulPartitionedCallв conv1d_5/StatefulPartitionedCallв conv1d_6/StatefulPartitionedCallв conv1d_7/StatefulPartitionedCallв conv1d_8/StatefulPartitionedCallв conv1d_9/StatefulPartitionedCall·
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_4260*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_27992 
conv1d/StatefulPartitionedCallб
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_4263batch_normalization_4265batch_normalization_4267batch_normalization_4269*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28462-
+batch_normalization/StatefulPartitionedCallЛ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_29072
activation/PartitionedCallа
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv1d_2_4273*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_29272"
 conv1d_2/StatefulPartitionedCall▓
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_1_4276batch_normalization_1_4278batch_normalization_1_4280batch_normalization_1_4282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29742/
-batch_normalization_1/StatefulPartitionedCallГ
max_pooling1d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_17332
max_pooling1d/PartitionedCallФ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_30362
activation_1/PartitionedCallв
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv1d_3_4287*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_30562"
 conv1d_3/StatefulPartitionedCallг
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_4290*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_30802"
 conv1d_1/StatefulPartitionedCallШ
add/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_30982
add/PartitionedCallе
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_2_4294batch_normalization_2_4296batch_normalization_2_4298batch_normalization_2_4300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31422/
-batch_normalization_2/StatefulPartitionedCallФ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_32032
activation_2/PartitionedCallв
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv1d_5_4304*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_5_layer_call_and_return_conditional_losses_32232"
 conv1d_5/StatefulPartitionedCall▓
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_4307batch_normalization_3_4309batch_normalization_3_4311batch_normalization_3_4313*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_32702/
-batch_normalization_3/StatefulPartitionedCallГ
max_pooling1d_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_20282!
max_pooling1d_1/PartitionedCallФ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_33322
activation_3/PartitionedCallв
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv1d_6_4318*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_6_layer_call_and_return_conditional_losses_33522"
 conv1d_6/StatefulPartitionedCallе
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_4_4321*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_4_layer_call_and_return_conditional_losses_33762"
 conv1d_4/StatefulPartitionedCallЮ
add_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33942
add_1/PartitionedCallз
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_4_4325batch_normalization_4_4327batch_normalization_4_4329batch_normalization_4_4331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34382/
-batch_normalization_4/StatefulPartitionedCallФ
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_34992
activation_4/PartitionedCallв
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv1d_8_4335*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_8_layer_call_and_return_conditional_losses_35192"
 conv1d_8/StatefulPartitionedCall▓
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0batch_normalization_5_4338batch_normalization_5_4340batch_normalization_5_4342batch_normalization_5_4344*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_35662/
-batch_normalization_5/StatefulPartitionedCallД
max_pooling1d_2/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23232!
max_pooling1d_2/PartitionedCallФ
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_36282
activation_5/PartitionedCallб
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv1d_9_4349*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_9_layer_call_and_return_conditional_losses_36482"
 conv1d_9/StatefulPartitionedCallд
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_7_4352*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_7_layer_call_and_return_conditional_losses_36722"
 conv1d_7/StatefulPartitionedCallЭ
add_2/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_2_layer_call_and_return_conditional_losses_36902
add_2/PartitionedCallж
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_6_4356batch_normalization_6_4358batch_normalization_6_4360batch_normalization_6_4362*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37342/
-batch_normalization_6/StatefulPartitionedCallУ
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_37952
activation_6/PartitionedCallе
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv1d_11_4366*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_11_layer_call_and_return_conditional_losses_38152#
!conv1d_11/StatefulPartitionedCall▓
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_7_4369batch_normalization_7_4371batch_normalization_7_4373batch_normalization_7_4375*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_38622/
-batch_normalization_7/StatefulPartitionedCallД
max_pooling1d_3/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_26182!
max_pooling1d_3/PartitionedCallУ
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_39242
activation_7/PartitionedCallе
!conv1d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv1d_12_4380*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_12_layer_call_and_return_conditional_losses_39442#
!conv1d_12/StatefulPartitionedCallи
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_3/PartitionedCall:output:0conv1d_10_4383*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_10_layer_call_and_return_conditional_losses_39682#
!conv1d_10/StatefulPartitionedCallЯ
add_3/PartitionedCallPartitionedCall*conv1d_12/StatefulPartitionedCall:output:0*conv1d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_3_layer_call_and_return_conditional_losses_39862
add_3/PartitionedCallж
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_8_4387batch_normalization_8_4389batch_normalization_8_4391batch_normalization_8_4393*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_40302/
-batch_normalization_8/StatefulPartitionedCallУ
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_40912
activation_8/PartitionedCallщ
embed/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_41042
embed/PartitionedCallщ
IdentityIdentityembed/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall"^conv1d_12/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2F
!conv1d_12/StatefulPartitionedCall!conv1d_12/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
┴
Р
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1573

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @:::::\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
В
Т
B__inference_conv1d_1_layer_call_and_return_conditional_losses_6183

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А@2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╕
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::T P
,
_output_shapes
:         А@
 
_user_specified_nameinputs
├
е
2__inference_batch_normalization_layer_call_fn_5867

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_28662
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
╞
@
$__inference_embed_layer_call_fn_7649

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_27802
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2008

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─:::::] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6848

inputs
assignmovingavg_6823
assignmovingavg_1_6829)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         АА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6823*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6823*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6823*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6823*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6823AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6823*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6829*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6829*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6829*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6829*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6829AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6829*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╨
m
'__inference_conv1d_6_layer_call_fn_6588

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_6_layer_call_and_return_conditional_losses_33522
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
Ф
@
$__inference_embed_layer_call_fn_7638

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_embed_layer_call_and_return_conditional_losses_41042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
Б
Т
B__inference_conv1d_7_layer_call_and_return_conditional_losses_7017

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @─::T P
,
_output_shapes
:         @─
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_3_layer_call_and_return_conditional_losses_3056

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А А::U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╚
b
F__inference_activation_7_layer_call_and_return_conditional_losses_7398

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         @└2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*+
_input_shapes
:         @└:T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6950

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7072

inputs
assignmovingavg_7047
assignmovingavg_1_7053)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         @А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7047*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7047*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7047*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7047*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7047AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7047*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7053*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7053*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7053*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7053*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7053AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7053*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
╬
n
(__inference_conv1d_10_layer_call_fn_7441

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         └*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_10_layer_call_and_return_conditional_losses_39682
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
щ
з
4__inference_batch_normalization_6_layer_call_fn_7187

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_24252
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
┐
g
=__inference_add_layer_call_and_return_conditional_losses_3098

inputs
inputs_1
identity]
addAddV2inputsinputs_1*
T0*-
_output_shapes
:         АА2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         АА:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         АА
 
_user_specified_nameinputs
ы)
┬
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1540

inputs
assignmovingavg_1515
assignmovingavg_1_1521)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/1515*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1515*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/1515*
_output_shapes
:@2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/1515*
_output_shapes
:@2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1515AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/1515*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/1521*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1521*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1521*
_output_shapes
:@2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1521*
_output_shapes
:@2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1521AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/1521*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╠
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6147

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А А2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А А:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2425

inputs
assignmovingavg_2400
assignmovingavg_1_2406)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/2400*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2400*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/2400*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/2400*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2400AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/2400*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/2406*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2406*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2406*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/2406*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2406AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/2406*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╠
b
F__inference_activation_2_layer_call_and_return_conditional_losses_6371

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         АА2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6737

inputs
assignmovingavg_6712
assignmovingavg_1_6718)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6712*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6712AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6718*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6718AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╦
з
4__inference_batch_normalization_1_layer_call_fn_6060

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29942
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╠
b
F__inference_activation_3_layer_call_and_return_conditional_losses_6564

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         А─2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3754

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А:::::T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6757

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─:::::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╬
m
'__inference_conv1d_9_layer_call_fn_7005

inputs
unknown
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_9_layer_call_and_return_conditional_losses_36482
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
▒
P
$__inference_add_1_layer_call_fn_6619
inputs_0
inputs_1
identity╨
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33942
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         А─:         А─:W S
-
_output_shapes
:         А─
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         А─
"
_user_specified_name
inputs/1
щ
з
4__inference_batch_normalization_5_layer_call_fn_6963

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_22702
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_2_layer_call_fn_6366

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18682
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
К
ч
+__inference_functional_1_layer_call_fn_5663

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*A
_read_only_resource_inputs#
!	
 !"%&'*+,-01*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_43992
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
╚
k
%__inference_conv1d_layer_call_fn_5785

inputs
unknown
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_27992
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А :22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
╦
з
4__inference_batch_normalization_4_layer_call_fn_6783

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34582
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6238

inputs
assignmovingavg_6213
assignmovingavg_1_6219)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         АА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6213*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6213*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6213*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6213*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6213AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6213*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6219*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6219*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6219*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6219*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6219AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6219*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2303

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╜═
 
__inference__wrapped_model_1444
ecgC
?functional_1_conv1d_conv1d_expanddims_1_readvariableop_resourceF
Bfunctional_1_batch_normalization_batchnorm_readvariableop_resourceJ
Ffunctional_1_batch_normalization_batchnorm_mul_readvariableop_resourceH
Dfunctional_1_batch_normalization_batchnorm_readvariableop_1_resourceH
Dfunctional_1_batch_normalization_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_2_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_1_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_1_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_1_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_1_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_3_conv1d_expanddims_1_readvariableop_resourceE
Afunctional_1_conv1d_1_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_2_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_2_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_2_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_2_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_5_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_3_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_3_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_3_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_3_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_6_conv1d_expanddims_1_readvariableop_resourceE
Afunctional_1_conv1d_4_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_4_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_4_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_4_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_4_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_8_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_5_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_5_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_5_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_5_batchnorm_readvariableop_2_resourceE
Afunctional_1_conv1d_9_conv1d_expanddims_1_readvariableop_resourceE
Afunctional_1_conv1d_7_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_6_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_6_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_6_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_6_batchnorm_readvariableop_2_resourceF
Bfunctional_1_conv1d_11_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_7_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_7_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_7_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_7_batchnorm_readvariableop_2_resourceF
Bfunctional_1_conv1d_12_conv1d_expanddims_1_readvariableop_resourceF
Bfunctional_1_conv1d_10_conv1d_expanddims_1_readvariableop_resourceH
Dfunctional_1_batch_normalization_8_batchnorm_readvariableop_resourceL
Hfunctional_1_batch_normalization_8_batchnorm_mul_readvariableop_resourceJ
Ffunctional_1_batch_normalization_8_batchnorm_readvariableop_1_resourceJ
Ffunctional_1_batch_normalization_8_batchnorm_readvariableop_2_resource
identityИб
)functional_1/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2+
)functional_1/conv1d/conv1d/ExpandDims/dim╨
%functional_1/conv1d/conv1d/ExpandDims
ExpandDimsecg2functional_1/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А 2'
%functional_1/conv1d/conv1d/ExpandDimsЇ
6functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?functional_1_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype028
6functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpЬ
+functional_1/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+functional_1/conv1d/conv1d/ExpandDims_1/dimЗ
'functional_1/conv1d/conv1d/ExpandDims_1
ExpandDims>functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:04functional_1/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2)
'functional_1/conv1d/conv1d/ExpandDims_1З
functional_1/conv1d/conv1dConv2D.functional_1/conv1d/conv1d/ExpandDims:output:00functional_1/conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         А @*
paddingSAME*
strides
2
functional_1/conv1d/conv1d╧
"functional_1/conv1d/conv1d/SqueezeSqueeze#functional_1/conv1d/conv1d:output:0*
T0*,
_output_shapes
:         А @*
squeeze_dims

¤        2$
"functional_1/conv1d/conv1d/Squeezeї
9functional_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOpBfunctional_1_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02;
9functional_1/batch_normalization/batchnorm/ReadVariableOpй
0functional_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0functional_1/batch_normalization/batchnorm/add/yМ
.functional_1/batch_normalization/batchnorm/addAddV2Afunctional_1/batch_normalization/batchnorm/ReadVariableOp:value:09functional_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@20
.functional_1/batch_normalization/batchnorm/add╞
0functional_1/batch_normalization/batchnorm/RsqrtRsqrt2functional_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@22
0functional_1/batch_normalization/batchnorm/RsqrtБ
=functional_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpFfunctional_1_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02?
=functional_1/batch_normalization/batchnorm/mul/ReadVariableOpЙ
.functional_1/batch_normalization/batchnorm/mulMul4functional_1/batch_normalization/batchnorm/Rsqrt:y:0Efunctional_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@20
.functional_1/batch_normalization/batchnorm/mulГ
0functional_1/batch_normalization/batchnorm/mul_1Mul+functional_1/conv1d/conv1d/Squeeze:output:02functional_1/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А @22
0functional_1/batch_normalization/batchnorm/mul_1√
;functional_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpDfunctional_1_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02=
;functional_1/batch_normalization/batchnorm/ReadVariableOp_1Й
0functional_1/batch_normalization/batchnorm/mul_2MulCfunctional_1/batch_normalization/batchnorm/ReadVariableOp_1:value:02functional_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@22
0functional_1/batch_normalization/batchnorm/mul_2√
;functional_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpDfunctional_1_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02=
;functional_1/batch_normalization/batchnorm/ReadVariableOp_2З
.functional_1/batch_normalization/batchnorm/subSubCfunctional_1/batch_normalization/batchnorm/ReadVariableOp_2:value:04functional_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@20
.functional_1/batch_normalization/batchnorm/subО
0functional_1/batch_normalization/batchnorm/add_1AddV24functional_1/batch_normalization/batchnorm/mul_1:z:02functional_1/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @22
0functional_1/batch_normalization/batchnorm/add_1▒
functional_1/activation/ReluRelu4functional_1/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         А @2
functional_1/activation/Reluе
+functional_1/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_2/conv1d/ExpandDims/dim¤
'functional_1/conv1d_2/conv1d/ExpandDims
ExpandDims*functional_1/activation/Relu:activations:04functional_1/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2)
'functional_1/conv1d_2/conv1d/ExpandDims√
8functional_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02:
8functional_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_2/conv1d/ExpandDims_1/dimР
)functional_1/conv1d_2/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2+
)functional_1/conv1d_2/conv1d/ExpandDims_1Р
functional_1/conv1d_2/conv1dConv2D0functional_1/conv1d_2/conv1d/ExpandDims:output:02functional_1/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А А*
paddingSAME*
strides
2
functional_1/conv1d_2/conv1d╓
$functional_1/conv1d_2/conv1d/SqueezeSqueeze%functional_1/conv1d_2/conv1d:output:0*
T0*-
_output_shapes
:         А А*
squeeze_dims

¤        2&
$functional_1/conv1d_2/conv1d/Squeeze№
;functional_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_1/batch_normalization_1/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_1/batchnorm/add/yХ
0functional_1/batch_normalization_1/batchnorm/addAddV2Cfunctional_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_1/batchnorm/add═
2functional_1/batch_normalization_1/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_1/batchnorm/RsqrtИ
?functional_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_1/batch_normalization_1/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_1/batchnorm/mulMul6functional_1/batch_normalization_1/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_1/batchnorm/mulМ
2functional_1/batch_normalization_1/batchnorm/mul_1Mul-functional_1/conv1d_2/conv1d/Squeeze:output:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А А24
2functional_1/batch_normalization_1/batchnorm/mul_1В
=functional_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_1/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_1/batchnorm/mul_2MulEfunctional_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_1/batchnorm/mul_2В
=functional_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_1/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_1/batchnorm/subSubEfunctional_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_1/batchnorm/subЧ
2functional_1/batch_normalization_1/batchnorm/add_1AddV26functional_1/batch_normalization_1/batchnorm/mul_1:z:04functional_1/batch_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А24
2functional_1/batch_normalization_1/batchnorm/add_1Ш
)functional_1/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/max_pooling1d/ExpandDims/dimў
%functional_1/max_pooling1d/ExpandDims
ExpandDims*functional_1/activation/Relu:activations:02functional_1/max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2'
%functional_1/max_pooling1d/ExpandDimsЁ
"functional_1/max_pooling1d/MaxPoolMaxPool.functional_1/max_pooling1d/ExpandDims:output:0*0
_output_shapes
:         А@*
ksize
*
paddingSAME*
strides
2$
"functional_1/max_pooling1d/MaxPool╬
"functional_1/max_pooling1d/SqueezeSqueeze+functional_1/max_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:         А@*
squeeze_dims
2$
"functional_1/max_pooling1d/Squeeze╕
functional_1/activation_1/ReluRelu6functional_1/batch_normalization_1/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А А2 
functional_1/activation_1/Reluе
+functional_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_3/conv1d/ExpandDims/dimА
'functional_1/conv1d_3/conv1d/ExpandDims
ExpandDims,functional_1/activation_1/Relu:activations:04functional_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А А2)
'functional_1/conv1d_3/conv1d/ExpandDims№
8functional_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02:
8functional_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_3/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_3/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2+
)functional_1/conv1d_3/conv1d/ExpandDims_1Р
functional_1/conv1d_3/conv1dConv2D0functional_1/conv1d_3/conv1d/ExpandDims:output:02functional_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
functional_1/conv1d_3/conv1d╓
$functional_1/conv1d_3/conv1d/SqueezeSqueeze%functional_1/conv1d_3/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2&
$functional_1/conv1d_3/conv1d/Squeezeе
+functional_1/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_1/conv1d/ExpandDims/dim■
'functional_1/conv1d_1/conv1d/ExpandDims
ExpandDims+functional_1/max_pooling1d/Squeeze:output:04functional_1/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А@2)
'functional_1/conv1d_1/conv1d/ExpandDims√
8functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02:
8functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_1/conv1d/ExpandDims_1/dimР
)functional_1/conv1d_1/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2+
)functional_1/conv1d_1/conv1d/ExpandDims_1Р
functional_1/conv1d_1/conv1dConv2D0functional_1/conv1d_1/conv1d/ExpandDims:output:02functional_1/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
functional_1/conv1d_1/conv1d╓
$functional_1/conv1d_1/conv1d/SqueezeSqueeze%functional_1/conv1d_1/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2&
$functional_1/conv1d_1/conv1d/Squeeze╦
functional_1/add/addAddV2-functional_1/conv1d_3/conv1d/Squeeze:output:0-functional_1/conv1d_1/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2
functional_1/add/add№
;functional_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_1/batch_normalization_2/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_2/batchnorm/add/yХ
0functional_1/batch_normalization_2/batchnorm/addAddV2Cfunctional_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_2/batchnorm/add═
2functional_1/batch_normalization_2/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_2/batchnorm/RsqrtИ
?functional_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_1/batch_normalization_2/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_2/batchnorm/mulMul6functional_1/batch_normalization_2/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_2/batchnorm/mulў
2functional_1/batch_normalization_2/batchnorm/mul_1Mulfunctional_1/add/add:z:04functional_1/batch_normalization_2/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА24
2functional_1/batch_normalization_2/batchnorm/mul_1В
=functional_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_2/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_2/batchnorm/mul_2MulEfunctional_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_2/batchnorm/mul_2В
=functional_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_2/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_2/batchnorm/subSubEfunctional_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_2/batchnorm/subЧ
2functional_1/batch_normalization_2/batchnorm/add_1AddV26functional_1/batch_normalization_2/batchnorm/mul_1:z:04functional_1/batch_normalization_2/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА24
2functional_1/batch_normalization_2/batchnorm/add_1╕
functional_1/activation_2/ReluRelu6functional_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2 
functional_1/activation_2/Reluе
+functional_1/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_5/conv1d/ExpandDims/dimА
'functional_1/conv1d_5/conv1d/ExpandDims
ExpandDims,functional_1/activation_2/Relu:activations:04functional_1/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2)
'functional_1/conv1d_5/conv1d/ExpandDims№
8functional_1/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_5_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02:
8functional_1/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_5/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_5/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2+
)functional_1/conv1d_5/conv1d/ExpandDims_1Р
functional_1/conv1d_5/conv1dConv2D0functional_1/conv1d_5/conv1d/ExpandDims:output:02functional_1/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
functional_1/conv1d_5/conv1d╓
$functional_1/conv1d_5/conv1d/SqueezeSqueeze%functional_1/conv1d_5/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2&
$functional_1/conv1d_5/conv1d/Squeeze№
;functional_1/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02=
;functional_1/batch_normalization_3/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_3/batchnorm/add/yХ
0functional_1/batch_normalization_3/batchnorm/addAddV2Cfunctional_1/batch_normalization_3/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_3/batchnorm/add═
2functional_1/batch_normalization_3/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:─24
2functional_1/batch_normalization_3/batchnorm/RsqrtИ
?functional_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02A
?functional_1/batch_normalization_3/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_3/batchnorm/mulMul6functional_1/batch_normalization_3/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_3/batchnorm/mulМ
2functional_1/batch_normalization_3/batchnorm/mul_1Mul-functional_1/conv1d_5/conv1d/Squeeze:output:04functional_1/batch_normalization_3/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─24
2functional_1/batch_normalization_3/batchnorm/mul_1В
=functional_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02?
=functional_1/batch_normalization_3/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_3/batchnorm/mul_2MulEfunctional_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:─24
2functional_1/batch_normalization_3/batchnorm/mul_2В
=functional_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02?
=functional_1/batch_normalization_3/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_3/batchnorm/subSubEfunctional_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_3/batchnorm/subЧ
2functional_1/batch_normalization_3/batchnorm/add_1AddV26functional_1/batch_normalization_3/batchnorm/mul_1:z:04functional_1/batch_normalization_3/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─24
2functional_1/batch_normalization_3/batchnorm/add_1Ь
+functional_1/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+functional_1/max_pooling1d_1/ExpandDims/dimь
'functional_1/max_pooling1d_1/ExpandDims
ExpandDimsfunctional_1/add/add:z:04functional_1/max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2)
'functional_1/max_pooling1d_1/ExpandDimsў
$functional_1/max_pooling1d_1/MaxPoolMaxPool0functional_1/max_pooling1d_1/ExpandDims:output:0*1
_output_shapes
:         АА*
ksize
*
paddingSAME*
strides
2&
$functional_1/max_pooling1d_1/MaxPool╒
$functional_1/max_pooling1d_1/SqueezeSqueeze-functional_1/max_pooling1d_1/MaxPool:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims
2&
$functional_1/max_pooling1d_1/Squeeze╕
functional_1/activation_3/ReluRelu6functional_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2 
functional_1/activation_3/Reluе
+functional_1/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_6/conv1d/ExpandDims/dimА
'functional_1/conv1d_6/conv1d/ExpandDims
ExpandDims,functional_1/activation_3/Relu:activations:04functional_1/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2)
'functional_1/conv1d_6/conv1d/ExpandDims№
8functional_1/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:──*
dtype02:
8functional_1/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_6/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_6/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:──2+
)functional_1/conv1d_6/conv1d/ExpandDims_1Р
functional_1/conv1d_6/conv1dConv2D0functional_1/conv1d_6/conv1d/ExpandDims:output:02functional_1/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
functional_1/conv1d_6/conv1d╓
$functional_1/conv1d_6/conv1d/SqueezeSqueeze%functional_1/conv1d_6/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2&
$functional_1/conv1d_6/conv1d/Squeezeе
+functional_1/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_4/conv1d/ExpandDims/dimБ
'functional_1/conv1d_4/conv1d/ExpandDims
ExpandDims-functional_1/max_pooling1d_1/Squeeze:output:04functional_1/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2)
'functional_1/conv1d_4/conv1d/ExpandDims№
8functional_1/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02:
8functional_1/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_4/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_4/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2+
)functional_1/conv1d_4/conv1d/ExpandDims_1Р
functional_1/conv1d_4/conv1dConv2D0functional_1/conv1d_4/conv1d/ExpandDims:output:02functional_1/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
functional_1/conv1d_4/conv1d╓
$functional_1/conv1d_4/conv1d/SqueezeSqueeze%functional_1/conv1d_4/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2&
$functional_1/conv1d_4/conv1d/Squeeze╧
functional_1/add_1/addAddV2-functional_1/conv1d_6/conv1d/Squeeze:output:0-functional_1/conv1d_4/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2
functional_1/add_1/add№
;functional_1/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02=
;functional_1/batch_normalization_4/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_4/batchnorm/add/yХ
0functional_1/batch_normalization_4/batchnorm/addAddV2Cfunctional_1/batch_normalization_4/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_4/batchnorm/add═
2functional_1/batch_normalization_4/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:─24
2functional_1/batch_normalization_4/batchnorm/RsqrtИ
?functional_1/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02A
?functional_1/batch_normalization_4/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_4/batchnorm/mulMul6functional_1/batch_normalization_4/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_4/batchnorm/mul∙
2functional_1/batch_normalization_4/batchnorm/mul_1Mulfunctional_1/add_1/add:z:04functional_1/batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─24
2functional_1/batch_normalization_4/batchnorm/mul_1В
=functional_1/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02?
=functional_1/batch_normalization_4/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_4/batchnorm/mul_2MulEfunctional_1/batch_normalization_4/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:─24
2functional_1/batch_normalization_4/batchnorm/mul_2В
=functional_1/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02?
=functional_1/batch_normalization_4/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_4/batchnorm/subSubEfunctional_1/batch_normalization_4/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─22
0functional_1/batch_normalization_4/batchnorm/subЧ
2functional_1/batch_normalization_4/batchnorm/add_1AddV26functional_1/batch_normalization_4/batchnorm/mul_1:z:04functional_1/batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─24
2functional_1/batch_normalization_4/batchnorm/add_1╕
functional_1/activation_4/ReluRelu6functional_1/batch_normalization_4/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2 
functional_1/activation_4/Reluе
+functional_1/conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_8/conv1d/ExpandDims/dimА
'functional_1/conv1d_8/conv1d/ExpandDims
ExpandDims,functional_1/activation_4/Relu:activations:04functional_1/conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2)
'functional_1/conv1d_8/conv1d/ExpandDims№
8functional_1/conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_8_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02:
8functional_1/conv1d_8/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_8/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_8/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2+
)functional_1/conv1d_8/conv1d/ExpandDims_1Р
functional_1/conv1d_8/conv1dConv2D0functional_1/conv1d_8/conv1d/ExpandDims:output:02functional_1/conv1d_8/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
functional_1/conv1d_8/conv1d╓
$functional_1/conv1d_8/conv1d/SqueezeSqueeze%functional_1/conv1d_8/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2&
$functional_1/conv1d_8/conv1d/Squeeze№
;functional_1/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_1/batch_normalization_5/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_5/batchnorm/add/yХ
0functional_1/batch_normalization_5/batchnorm/addAddV2Cfunctional_1/batch_normalization_5/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_5/batchnorm/add═
2functional_1/batch_normalization_5/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_5/batchnorm/RsqrtИ
?functional_1/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_1/batch_normalization_5/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_5/batchnorm/mulMul6functional_1/batch_normalization_5/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_5/batchnorm/mulМ
2functional_1/batch_normalization_5/batchnorm/mul_1Mul-functional_1/conv1d_8/conv1d/Squeeze:output:04functional_1/batch_normalization_5/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА24
2functional_1/batch_normalization_5/batchnorm/mul_1В
=functional_1/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_5/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_5/batchnorm/mul_2MulEfunctional_1/batch_normalization_5/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_5/batchnorm/mul_2В
=functional_1/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_5/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_5/batchnorm/subSubEfunctional_1/batch_normalization_5/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_5/batchnorm/subЧ
2functional_1/batch_normalization_5/batchnorm/add_1AddV26functional_1/batch_normalization_5/batchnorm/mul_1:z:04functional_1/batch_normalization_5/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА24
2functional_1/batch_normalization_5/batchnorm/add_1Ь
+functional_1/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+functional_1/max_pooling1d_2/ExpandDims/dimю
'functional_1/max_pooling1d_2/ExpandDims
ExpandDimsfunctional_1/add_1/add:z:04functional_1/max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2)
'functional_1/max_pooling1d_2/ExpandDimsЎ
$functional_1/max_pooling1d_2/MaxPoolMaxPool0functional_1/max_pooling1d_2/ExpandDims:output:0*0
_output_shapes
:         @─*
ksize
*
paddingSAME*
strides
2&
$functional_1/max_pooling1d_2/MaxPool╘
$functional_1/max_pooling1d_2/SqueezeSqueeze-functional_1/max_pooling1d_2/MaxPool:output:0*
T0*,
_output_shapes
:         @─*
squeeze_dims
2&
$functional_1/max_pooling1d_2/Squeeze╕
functional_1/activation_5/ReluRelu6functional_1/batch_normalization_5/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2 
functional_1/activation_5/Reluе
+functional_1/conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_9/conv1d/ExpandDims/dimА
'functional_1/conv1d_9/conv1d/ExpandDims
ExpandDims,functional_1/activation_5/Relu:activations:04functional_1/conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2)
'functional_1/conv1d_9/conv1d/ExpandDims№
8functional_1/conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_9_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02:
8functional_1/conv1d_9/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_9/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_9/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2+
)functional_1/conv1d_9/conv1d/ExpandDims_1П
functional_1/conv1d_9/conv1dConv2D0functional_1/conv1d_9/conv1d/ExpandDims:output:02functional_1/conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
functional_1/conv1d_9/conv1d╒
$functional_1/conv1d_9/conv1d/SqueezeSqueeze%functional_1/conv1d_9/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2&
$functional_1/conv1d_9/conv1d/Squeezeе
+functional_1/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2-
+functional_1/conv1d_7/conv1d/ExpandDims/dimА
'functional_1/conv1d_7/conv1d/ExpandDims
ExpandDims-functional_1/max_pooling1d_2/Squeeze:output:04functional_1/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @─2)
'functional_1/conv1d_7/conv1d/ExpandDims№
8functional_1/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02:
8functional_1/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpа
-functional_1/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_7/conv1d/ExpandDims_1/dimС
)functional_1/conv1d_7/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2+
)functional_1/conv1d_7/conv1d/ExpandDims_1П
functional_1/conv1d_7/conv1dConv2D0functional_1/conv1d_7/conv1d/ExpandDims:output:02functional_1/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
functional_1/conv1d_7/conv1d╒
$functional_1/conv1d_7/conv1d/SqueezeSqueeze%functional_1/conv1d_7/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2&
$functional_1/conv1d_7/conv1d/Squeeze╬
functional_1/add_2/addAddV2-functional_1/conv1d_9/conv1d/Squeeze:output:0-functional_1/conv1d_7/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2
functional_1/add_2/add№
;functional_1/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;functional_1/batch_normalization_6/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_6/batchnorm/add/yХ
0functional_1/batch_normalization_6/batchnorm/addAddV2Cfunctional_1/batch_normalization_6/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_6/batchnorm/add═
2functional_1/batch_normalization_6/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_6/batchnorm/RsqrtИ
?functional_1/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?functional_1/batch_normalization_6/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_6/batchnorm/mulMul6functional_1/batch_normalization_6/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_6/batchnorm/mul°
2functional_1/batch_normalization_6/batchnorm/mul_1Mulfunctional_1/add_2/add:z:04functional_1/batch_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @А24
2functional_1/batch_normalization_6/batchnorm/mul_1В
=functional_1/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_6/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_6/batchnorm/mul_2MulEfunctional_1/batch_normalization_6/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А24
2functional_1/batch_normalization_6/batchnorm/mul_2В
=functional_1/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02?
=functional_1/batch_normalization_6/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_6/batchnorm/subSubEfunctional_1/batch_normalization_6/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А22
0functional_1/batch_normalization_6/batchnorm/subЦ
2functional_1/batch_normalization_6/batchnorm/add_1AddV26functional_1/batch_normalization_6/batchnorm/mul_1:z:04functional_1/batch_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А24
2functional_1/batch_normalization_6/batchnorm/add_1╖
functional_1/activation_6/ReluRelu6functional_1/batch_normalization_6/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @А2 
functional_1/activation_6/Reluз
,functional_1/conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2.
,functional_1/conv1d_11/conv1d/ExpandDims/dimВ
(functional_1/conv1d_11/conv1d/ExpandDims
ExpandDims,functional_1/activation_6/Relu:activations:05functional_1/conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2*
(functional_1/conv1d_11/conv1d/ExpandDims 
9functional_1/conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBfunctional_1_conv1d_11_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02;
9functional_1/conv1d_11/conv1d/ExpandDims_1/ReadVariableOpв
.functional_1/conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/conv1d_11/conv1d/ExpandDims_1/dimХ
*functional_1/conv1d_11/conv1d/ExpandDims_1
ExpandDimsAfunctional_1/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:07functional_1/conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2,
*functional_1/conv1d_11/conv1d/ExpandDims_1У
functional_1/conv1d_11/conv1dConv2D1functional_1/conv1d_11/conv1d/ExpandDims:output:03functional_1/conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @└*
paddingSAME*
strides
2
functional_1/conv1d_11/conv1d╪
%functional_1/conv1d_11/conv1d/SqueezeSqueeze&functional_1/conv1d_11/conv1d:output:0*
T0*,
_output_shapes
:         @└*
squeeze_dims

¤        2'
%functional_1/conv1d_11/conv1d/Squeeze№
;functional_1/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02=
;functional_1/batch_normalization_7/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_7/batchnorm/add/yХ
0functional_1/batch_normalization_7/batchnorm/addAddV2Cfunctional_1/batch_normalization_7/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_7/batchnorm/add═
2functional_1/batch_normalization_7/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:└24
2functional_1/batch_normalization_7/batchnorm/RsqrtИ
?functional_1/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02A
?functional_1/batch_normalization_7/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_7/batchnorm/mulMul6functional_1/batch_normalization_7/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_7/batchnorm/mulМ
2functional_1/batch_normalization_7/batchnorm/mul_1Mul.functional_1/conv1d_11/conv1d/Squeeze:output:04functional_1/batch_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @└24
2functional_1/batch_normalization_7/batchnorm/mul_1В
=functional_1/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02?
=functional_1/batch_normalization_7/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_7/batchnorm/mul_2MulEfunctional_1/batch_normalization_7/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:└24
2functional_1/batch_normalization_7/batchnorm/mul_2В
=functional_1/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02?
=functional_1/batch_normalization_7/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_7/batchnorm/subSubEfunctional_1/batch_normalization_7/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_7/batchnorm/subЦ
2functional_1/batch_normalization_7/batchnorm/add_1AddV26functional_1/batch_normalization_7/batchnorm/mul_1:z:04functional_1/batch_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└24
2functional_1/batch_normalization_7/batchnorm/add_1Ь
+functional_1/max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+functional_1/max_pooling1d_3/ExpandDims/dimэ
'functional_1/max_pooling1d_3/ExpandDims
ExpandDimsfunctional_1/add_2/add:z:04functional_1/max_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2)
'functional_1/max_pooling1d_3/ExpandDimsЎ
$functional_1/max_pooling1d_3/MaxPoolMaxPool0functional_1/max_pooling1d_3/ExpandDims:output:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2&
$functional_1/max_pooling1d_3/MaxPool╘
$functional_1/max_pooling1d_3/SqueezeSqueeze-functional_1/max_pooling1d_3/MaxPool:output:0*
T0*,
_output_shapes
:         А*
squeeze_dims
2&
$functional_1/max_pooling1d_3/Squeeze╖
functional_1/activation_7/ReluRelu6functional_1/batch_normalization_7/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @└2 
functional_1/activation_7/Reluз
,functional_1/conv1d_12/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2.
,functional_1/conv1d_12/conv1d/ExpandDims/dimВ
(functional_1/conv1d_12/conv1d/ExpandDims
ExpandDims,functional_1/activation_7/Relu:activations:05functional_1/conv1d_12/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @└2*
(functional_1/conv1d_12/conv1d/ExpandDims 
9functional_1/conv1d_12/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBfunctional_1_conv1d_12_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:└└*
dtype02;
9functional_1/conv1d_12/conv1d/ExpandDims_1/ReadVariableOpв
.functional_1/conv1d_12/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/conv1d_12/conv1d/ExpandDims_1/dimХ
*functional_1/conv1d_12/conv1d/ExpandDims_1
ExpandDimsAfunctional_1/conv1d_12/conv1d/ExpandDims_1/ReadVariableOp:value:07functional_1/conv1d_12/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:└└2,
*functional_1/conv1d_12/conv1d/ExpandDims_1У
functional_1/conv1d_12/conv1dConv2D1functional_1/conv1d_12/conv1d/ExpandDims:output:03functional_1/conv1d_12/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
functional_1/conv1d_12/conv1d╪
%functional_1/conv1d_12/conv1d/SqueezeSqueeze&functional_1/conv1d_12/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2'
%functional_1/conv1d_12/conv1d/Squeezeз
,functional_1/conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2.
,functional_1/conv1d_10/conv1d/ExpandDims/dimГ
(functional_1/conv1d_10/conv1d/ExpandDims
ExpandDims-functional_1/max_pooling1d_3/Squeeze:output:05functional_1/conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2*
(functional_1/conv1d_10/conv1d/ExpandDims 
9functional_1/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBfunctional_1_conv1d_10_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02;
9functional_1/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpв
.functional_1/conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/conv1d_10/conv1d/ExpandDims_1/dimХ
*functional_1/conv1d_10/conv1d/ExpandDims_1
ExpandDimsAfunctional_1/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:07functional_1/conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2,
*functional_1/conv1d_10/conv1d/ExpandDims_1У
functional_1/conv1d_10/conv1dConv2D1functional_1/conv1d_10/conv1d/ExpandDims:output:03functional_1/conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
functional_1/conv1d_10/conv1d╪
%functional_1/conv1d_10/conv1d/SqueezeSqueeze&functional_1/conv1d_10/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2'
%functional_1/conv1d_10/conv1d/Squeeze╨
functional_1/add_3/addAddV2.functional_1/conv1d_12/conv1d/Squeeze:output:0.functional_1/conv1d_10/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2
functional_1/add_3/add№
;functional_1/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOpDfunctional_1_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02=
;functional_1/batch_normalization_8/batchnorm/ReadVariableOpн
2functional_1/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2functional_1/batch_normalization_8/batchnorm/add/yХ
0functional_1/batch_normalization_8/batchnorm/addAddV2Cfunctional_1/batch_normalization_8/batchnorm/ReadVariableOp:value:0;functional_1/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_8/batchnorm/add═
2functional_1/batch_normalization_8/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:└24
2functional_1/batch_normalization_8/batchnorm/RsqrtИ
?functional_1/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02A
?functional_1/batch_normalization_8/batchnorm/mul/ReadVariableOpТ
0functional_1/batch_normalization_8/batchnorm/mulMul6functional_1/batch_normalization_8/batchnorm/Rsqrt:y:0Gfunctional_1/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_8/batchnorm/mul°
2functional_1/batch_normalization_8/batchnorm/mul_1Mulfunctional_1/add_3/add:z:04functional_1/batch_normalization_8/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └24
2functional_1/batch_normalization_8/batchnorm/mul_1В
=functional_1/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpFfunctional_1_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02?
=functional_1/batch_normalization_8/batchnorm/ReadVariableOp_1Т
2functional_1/batch_normalization_8/batchnorm/mul_2MulEfunctional_1/batch_normalization_8/batchnorm/ReadVariableOp_1:value:04functional_1/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:└24
2functional_1/batch_normalization_8/batchnorm/mul_2В
=functional_1/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpFfunctional_1_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02?
=functional_1/batch_normalization_8/batchnorm/ReadVariableOp_2Р
0functional_1/batch_normalization_8/batchnorm/subSubEfunctional_1/batch_normalization_8/batchnorm/ReadVariableOp_2:value:06functional_1/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└22
0functional_1/batch_normalization_8/batchnorm/subЦ
2functional_1/batch_normalization_8/batchnorm/add_1AddV26functional_1/batch_normalization_8/batchnorm/mul_1:z:04functional_1/batch_normalization_8/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └24
2functional_1/batch_normalization_8/batchnorm/add_1╖
functional_1/activation_8/ReluRelu6functional_1/batch_normalization_8/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └2 
functional_1/activation_8/ReluШ
)functional_1/embed/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/embed/Mean/reduction_indices╧
functional_1/embed/MeanMean,functional_1/activation_8/Relu:activations:02functional_1/embed/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         └2
functional_1/embed/Meanu
IdentityIdentity functional_1/embed/Mean:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А ::::::::::::::::::::::::::::::::::::::::::::::::::Q M
,
_output_shapes
:         А 

_user_specified_nameecg
Б
Т
B__inference_conv1d_7_layer_call_and_return_conditional_losses_3672

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @─::T P
,
_output_shapes
:         @─
 
_user_specified_nameinputs
ё
H
,__inference_max_pooling1d_layer_call_fn_1739

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_17332
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
├
k
?__inference_add_3_layer_call_and_return_conditional_losses_7447
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:         └2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         └:         └:V R
,
_output_shapes
:         └
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         └
"
_user_specified_name
inputs/1
┼
з
4__inference_batch_normalization_6_layer_call_fn_7105

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37342
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
В
У
C__inference_conv1d_10_layer_call_and_return_conditional_losses_7434

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
с
е
2__inference_batch_normalization_layer_call_fn_5936

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_15402
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6431

inputs
assignmovingavg_6406
assignmovingavg_1_6412)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:─2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  ─2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6406*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6406*
_output_shapes	
:─*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6406*
_output_shapes	
:─2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6406*
_output_shapes	
:─2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6406AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6406*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6412*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6412*
_output_shapes	
:─*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6412*
_output_shapes	
:─2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6412*
_output_shapes	
:─2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6412AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6412*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  ─2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  ─2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  ─::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  ─
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7571

inputs
assignmovingavg_7546
assignmovingavg_1_7552)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7546*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7546*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7546*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7546*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7546AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7546*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7552*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7552*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7552*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7552*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7552AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7552*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  └2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_6_layer_call_and_return_conditional_losses_6581

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:──*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:──2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7285

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└:::::T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
щ
з
4__inference_batch_normalization_7_layer_call_fn_7380

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_25652
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╨
m
'__inference_conv1d_5_layer_call_fn_6395

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_5_layer_call_and_return_conditional_losses_32232
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
ы)
┬
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5903

inputs
assignmovingavg_5878
assignmovingavg_1_5884)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/5878*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayС
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5878*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp┴
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/5878*
_output_shapes
:@2
AssignMovingAvg/sub╕
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/5878*
_output_shapes
:@2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5878AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/5878*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/5884*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЧ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5884*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╦
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5884*
_output_shapes
:@2
AssignMovingAvg_1/sub┬
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5884*
_output_shapes
:@2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5884AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/5884*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╠
b
F__inference_activation_2_layer_call_and_return_conditional_losses_3203

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         АА2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3162

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА:::::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
╔
з
4__inference_batch_normalization_1_layer_call_fn_6047

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29742
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╟
з
4__inference_batch_normalization_6_layer_call_fn_7118

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37542
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1680

inputs
assignmovingavg_1655
assignmovingavg_1_1661)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/1655*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1655*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/1655*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/1655*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1655AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/1655*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/1661*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1661*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1661*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/1661*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1661AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/1661*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ї
J
.__inference_max_pooling1d_3_layer_call_fn_2624

inputs
identity▌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_26182
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
х
e
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2028

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
л
P
$__inference_add_2_layer_call_fn_7036
inputs_0
inputs_1
identity╧
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_add_2_layer_call_and_return_conditional_losses_36902
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         @А:         @А:V R
,
_output_shapes
:         @А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         @А
"
_user_specified_name
inputs/1
╠
m
'__inference_conv1d_7_layer_call_fn_7024

inputs
unknown
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         @А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_7_layer_call_and_return_conditional_losses_36722
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         @─:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         @─
 
_user_specified_nameinputs
щ
з
4__inference_batch_normalization_2_layer_call_fn_6353

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18352
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6533

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─:::::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_1_layer_call_fn_6142

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17132
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3290

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:─2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:─2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:─*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А─:::::U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
и
Т
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6258

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         АА:::::U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
о
G
+__inference_activation_4_layer_call_fn_6793

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_34992
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*,
_input_shapes
:         А─:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7174

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
х
e
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2323

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims░
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingSAME*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
√
[
?__inference_embed_layer_call_and_return_conditional_losses_2780

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Б
ф
+__inference_functional_1_layer_call_fn_4500
ecg
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallecgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*A
_read_only_resource_inputs#
!	
 !"%&'*+,-01*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_43992
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
,
_output_shapes
:         А 

_user_specified_nameecg
╨
m
'__inference_conv1d_3_layer_call_fn_6171

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_3_layer_call_and_return_conditional_losses_30562
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А А:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7092

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @А:::::T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs
╗
i
?__inference_add_2_layer_call_and_return_conditional_losses_3690

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:         @А2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:         @А2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:         @А:         @А:T P
,
_output_shapes
:         @А
 
_user_specified_nameinputs:TP
,
_output_shapes
:         @А
 
_user_specified_nameinputs
З
Т
B__inference_conv1d_3_layer_call_and_return_conditional_losses_6164

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А А::U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╟
i
=__inference_add_layer_call_and_return_conditional_losses_6196
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:         АА2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         АА:         АА:W S
-
_output_shapes
:         АА
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         АА
"
_user_specified_name
inputs/1
╚
b
F__inference_activation_8_layer_call_and_return_conditional_losses_7622

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         └2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
В
У
C__inference_conv1d_10_layer_call_and_return_conditional_losses_3968

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╬)
─
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7489

inputs
assignmovingavg_7464
assignmovingavg_1_7470)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:└2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         └2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7464*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7464*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7464*
_output_shapes	
:└2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7464*
_output_shapes	
:└2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7464AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7464*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7470*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7470*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7470*
_output_shapes	
:└2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7470*
_output_shapes	
:└2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7470AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7470*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         └2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
╤
Т
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1868

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А:::::] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
┴
Р
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5923

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @:::::\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╔
[
?__inference_embed_layer_call_and_return_conditional_losses_7633

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         └2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
ы
з
4__inference_batch_normalization_8_layer_call_fn_7617

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  └*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27532
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  └2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  └::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  └
 
_user_specified_nameinputs
╚├
Ў
F__inference_functional_1_layer_call_and_return_conditional_losses_5277

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource,
(batch_normalization_assignmovingavg_4869.
*batch_normalization_assignmovingavg_1_4875=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_1_assignmovingavg_49100
,batch_normalization_1_assignmovingavg_1_4916?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_2_assignmovingavg_49640
,batch_normalization_2_assignmovingavg_1_4970?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_3_assignmovingavg_50050
,batch_normalization_3_assignmovingavg_1_5011?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_4_assignmovingavg_50590
,batch_normalization_4_assignmovingavg_1_5065?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource8
4conv1d_8_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_5_assignmovingavg_51000
,batch_normalization_5_assignmovingavg_1_5106?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource8
4conv1d_9_conv1d_expanddims_1_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_6_assignmovingavg_51540
,batch_normalization_6_assignmovingavg_1_5160?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_7_assignmovingavg_51950
,batch_normalization_7_assignmovingavg_1_5201?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource9
5conv1d_12_conv1d_expanddims_1_readvariableop_resource9
5conv1d_10_conv1d_expanddims_1_readvariableop_resource.
*batch_normalization_8_assignmovingavg_52490
,batch_normalization_8_assignmovingavg_1_5255?
;batch_normalization_8_batchnorm_mul_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource
identityИв7batch_normalization/AssignMovingAvg/AssignSubVariableOpв9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimм
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А 2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/conv1d/ExpandDims_1╙
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         А @*
paddingSAME*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         А @*
squeeze_dims

¤        2
conv1d/conv1d/Squeeze╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesч
 batch_normalization/moments/meanMeanconv1d/conv1d/Squeeze:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradient¤
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv1d/conv1d/Squeeze:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         А @2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesЖ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1╪
)batch_normalization/AssignMovingAvg/decayConst*;
_class1
/-loc:@batch_normalization/AssignMovingAvg/4869*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2+
)batch_normalization/AssignMovingAvg/decay═
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp(batch_normalization_assignmovingavg_4869*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpе
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*;
_class1
/-loc:@batch_normalization/AssignMovingAvg/4869*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/subЬ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*;
_class1
/-loc:@batch_normalization/AssignMovingAvg/4869*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulї
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(batch_normalization_assignmovingavg_4869+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*;
_class1
/-loc:@batch_normalization/AssignMovingAvg/4869*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp▐
+batch_normalization/AssignMovingAvg_1/decayConst*=
_class3
1/loc:@batch_normalization/AssignMovingAvg_1/4875*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization/AssignMovingAvg_1/decay╙
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_1_4875*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpп
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg_1/4875*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subж
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg_1/4875*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mulБ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_1_4875-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*=
_class3
1/loc:@batch_normalization/AssignMovingAvg_1/4875*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╥
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul╧
#batch_normalization/batchnorm/mul_1Mulconv1d/conv1d/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         А @2
activation/ReluЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╔
conv1d_2/conv1d/ExpandDims
ExpandDimsactivation/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
conv1d_2/conv1d/ExpandDims╘
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim▄
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А А*
paddingSAME*
strides
2
conv1d_2/conv1dп
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*-
_output_shapes
:         А А*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeeze╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesЁ
"batch_normalization_1/moments/meanMean conv1d_2/conv1d/Squeeze:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2$
"batch_normalization_1/moments/mean├
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:А2,
*batch_normalization_1/moments/StopGradientЖ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference conv1d_2/conv1d/Squeeze:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:         А А21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesП
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2(
&batch_normalization_1/moments/variance─
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╠
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1▐
+batch_normalization_1/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/4910*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decay╘
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_1_assignmovingavg_4910*
_output_shapes	
:А*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp░
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/4910*
_output_shapes	
:А2+
)batch_normalization_1/AssignMovingAvg/subз
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/4910*
_output_shapes	
:А2+
)batch_normalization_1/AssignMovingAvg/mulБ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_1_assignmovingavg_4910-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/4910*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_1/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/4916*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decay┌
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_1_4916*
_output_shapes	
:А*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/4916*
_output_shapes	
:А2-
+batch_normalization_1/AssignMovingAvg_1/sub▒
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/4916*
_output_shapes	
:А2-
+batch_normalization_1/AssignMovingAvg_1/mulН
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_1_4916/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/4916*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/addж
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_1/batchnorm/Rsqrtс
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▐
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/mul╪
%batch_normalization_1/batchnorm/mul_1Mul conv1d_2/conv1d/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2'
%batch_normalization_1/batchnorm/mul_1╘
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_1/batchnorm/mul_2╒
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┌
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_1/batchnorm/subу
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2'
%batch_normalization_1/batchnorm/add_1~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim├
max_pooling1d/ExpandDims
ExpandDimsactivation/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А @2
max_pooling1d/ExpandDims╔
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:         А@*
ksize
*
paddingSAME*
strides
2
max_pooling1d/MaxPoolз
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:         А@*
squeeze_dims
2
max_pooling1d/SqueezeС
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А А2
activation_1/ReluЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_3/conv1d/ExpandDims/dim╠
conv1d_3/conv1d/ExpandDims
ExpandDimsactivation_1/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А А2
conv1d_3/conv1d/ExpandDims╒
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim▌
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d_3/conv1d/ExpandDims_1▄
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_3/conv1dп
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_3/conv1d/SqueezeЛ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╩
conv1d_1/conv1d/ExpandDims
ExpandDimsmax_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А@2
conv1d_1/conv1d/ExpandDims╘
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim▄
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_1/conv1dп
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_1/conv1d/SqueezeЧ
add/addAddV2 conv1d_3/conv1d/Squeeze:output:0 conv1d_1/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2	
add/add╜
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices█
"batch_normalization_2/moments/meanMeanadd/add:z:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2$
"batch_normalization_2/moments/mean├
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*#
_output_shapes
:А2,
*batch_normalization_2/moments/StopGradientё
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03batch_normalization_2/moments/StopGradient:output:0*
T0*-
_output_shapes
:         АА21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesП
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2(
&batch_normalization_2/moments/variance─
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╠
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1▐
+batch_normalization_2/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/4964*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_2/AssignMovingAvg/decay╘
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_2_assignmovingavg_4964*
_output_shapes	
:А*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp░
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/4964*
_output_shapes	
:А2+
)batch_normalization_2/AssignMovingAvg/subз
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/4964*
_output_shapes	
:А2+
)batch_normalization_2/AssignMovingAvg/mulБ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_2_assignmovingavg_4964-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/4964*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_2/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/4970*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decay┌
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_1_4970*
_output_shapes	
:А*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/4970*
_output_shapes	
:А2-
+batch_normalization_2/AssignMovingAvg_1/sub▒
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/4970*
_output_shapes	
:А2-
+batch_normalization_2/AssignMovingAvg_1/mulН
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_1_4970/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/4970*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/y█
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/addж
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_2/batchnorm/Rsqrtс
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▐
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/mul├
%batch_normalization_2/batchnorm/mul_1Muladd/add:z:0'batch_normalization_2/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_2/batchnorm/mul_1╘
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_2/batchnorm/mul_2╒
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┌
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_2/batchnorm/subу
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_2/batchnorm/add_1С
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2
activation_2/ReluЛ
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_5/conv1d/ExpandDims/dim╠
conv1d_5/conv1d/ExpandDims
ExpandDimsactivation_2/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_5/conv1d/ExpandDims╒
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim▌
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d_5/conv1d/ExpandDims_1▄
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_5/conv1dп
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_5/conv1d/Squeeze╜
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_3/moments/mean/reduction_indicesЁ
"batch_normalization_3/moments/meanMean conv1d_5/conv1d/Squeeze:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2$
"batch_normalization_3/moments/mean├
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*#
_output_shapes
:─2,
*batch_normalization_3/moments/StopGradientЖ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifference conv1d_5/conv1d/Squeeze:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*-
_output_shapes
:         А─21
/batch_normalization_3/moments/SquaredDifference┼
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_3/moments/variance/reduction_indicesП
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2(
&batch_normalization_3/moments/variance─
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╠
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1▐
+batch_normalization_3/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/5005*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_3/AssignMovingAvg/decay╘
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_3_assignmovingavg_5005*
_output_shapes	
:─*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp░
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/5005*
_output_shapes	
:─2+
)batch_normalization_3/AssignMovingAvg/subз
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/5005*
_output_shapes	
:─2+
)batch_normalization_3/AssignMovingAvg/mulБ
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_3_assignmovingavg_5005-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/5005*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_3/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/5011*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_3/AssignMovingAvg_1/decay┌
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_1_5011*
_output_shapes	
:─*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/5011*
_output_shapes	
:─2-
+batch_normalization_3/AssignMovingAvg_1/sub▒
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/5011*
_output_shapes	
:─2-
+batch_normalization_3/AssignMovingAvg_1/mulН
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_1_5011/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/5011*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/y█
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/addж
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_3/batchnorm/Rsqrtс
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▐
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/mul╪
%batch_normalization_3/batchnorm/mul_1Mul conv1d_5/conv1d/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_3/batchnorm/mul_1╘
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_3/batchnorm/mul_2╒
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┌
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2%
#batch_normalization_3/batchnorm/subу
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_3/batchnorm/add_1В
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim╕
max_pooling1d_1/ExpandDims
ExpandDimsadd/add:z:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
max_pooling1d_1/ExpandDims╨
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*1
_output_shapes
:         АА*
ksize
*
paddingSAME*
strides
2
max_pooling1d_1/MaxPoolо
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims
2
max_pooling1d_1/SqueezeС
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2
activation_3/ReluЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_6/conv1d/ExpandDims/dim╠
conv1d_6/conv1d/ExpandDims
ExpandDimsactivation_3/Relu:activations:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d_6/conv1d/ExpandDims╒
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:──*
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim▌
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:──2
conv1d_6/conv1d/ExpandDims_1▄
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_6/conv1dп
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_6/conv1d/SqueezeЛ
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_4/conv1d/ExpandDims/dim═
conv1d_4/conv1d/ExpandDims
ExpandDims max_pooling1d_1/Squeeze:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_4/conv1d/ExpandDims╒
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А─*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim▌
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А─2
conv1d_4/conv1d/ExpandDims_1▄
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         А─*
paddingSAME*
strides
2
conv1d_4/conv1dп
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*-
_output_shapes
:         А─*
squeeze_dims

¤        2
conv1d_4/conv1d/SqueezeЫ
	add_1/addAddV2 conv1d_6/conv1d/Squeeze:output:0 conv1d_4/conv1d/Squeeze:output:0*
T0*-
_output_shapes
:         А─2
	add_1/add╜
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_4/moments/mean/reduction_indices▌
"batch_normalization_4/moments/meanMeanadd_1/add:z:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2$
"batch_normalization_4/moments/mean├
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*#
_output_shapes
:─2,
*batch_normalization_4/moments/StopGradientє
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03batch_normalization_4/moments/StopGradient:output:0*
T0*-
_output_shapes
:         А─21
/batch_normalization_4/moments/SquaredDifference┼
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_4/moments/variance/reduction_indicesП
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:─*
	keep_dims(2(
&batch_normalization_4/moments/variance─
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╠
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:─*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1▐
+batch_normalization_4/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/5059*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_4/AssignMovingAvg/decay╘
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_4_assignmovingavg_5059*
_output_shapes	
:─*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp░
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/5059*
_output_shapes	
:─2+
)batch_normalization_4/AssignMovingAvg/subз
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/5059*
_output_shapes	
:─2+
)batch_normalization_4/AssignMovingAvg/mulБ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_4_assignmovingavg_5059-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/5059*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_4/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/5065*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_4/AssignMovingAvg_1/decay┌
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_1_5065*
_output_shapes	
:─*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/5065*
_output_shapes	
:─2-
+batch_normalization_4/AssignMovingAvg_1/sub▒
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/5065*
_output_shapes	
:─2-
+batch_normalization_4/AssignMovingAvg_1/mulН
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_1_5065/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/5065*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/addж
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_4/batchnorm/Rsqrtс
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:─*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp▐
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/mul┼
%batch_normalization_4/batchnorm/mul_1Muladd_1/add:z:0'batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_4/batchnorm/mul_1╘
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:─2'
%batch_normalization_4/batchnorm/mul_2╒
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:─*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp┌
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:─2%
#batch_normalization_4/batchnorm/subу
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:         А─2'
%batch_normalization_4/batchnorm/add_1С
activation_4/ReluRelu)batch_normalization_4/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         А─2
activation_4/ReluЛ
conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_8/conv1d/ExpandDims/dim╠
conv1d_8/conv1d/ExpandDims
ExpandDimsactivation_4/Relu:activations:0'conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
conv1d_8/conv1d/ExpandDims╒
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02-
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_8/conv1d/ExpandDims_1/dim▌
conv1d_8/conv1d/ExpandDims_1
ExpandDims3conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d_8/conv1d/ExpandDims_1▄
conv1d_8/conv1dConv2D#conv1d_8/conv1d/ExpandDims:output:0%conv1d_8/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1d_8/conv1dп
conv1d_8/conv1d/SqueezeSqueezeconv1d_8/conv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d_8/conv1d/Squeeze╜
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_5/moments/mean/reduction_indicesЁ
"batch_normalization_5/moments/meanMean conv1d_8/conv1d/Squeeze:output:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2$
"batch_normalization_5/moments/mean├
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*#
_output_shapes
:А2,
*batch_normalization_5/moments/StopGradientЖ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifference conv1d_8/conv1d/Squeeze:output:03batch_normalization_5/moments/StopGradient:output:0*
T0*-
_output_shapes
:         АА21
/batch_normalization_5/moments/SquaredDifference┼
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_5/moments/variance/reduction_indicesП
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2(
&batch_normalization_5/moments/variance─
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze╠
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1▐
+batch_normalization_5/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/5100*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_5/AssignMovingAvg/decay╘
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_5_assignmovingavg_5100*
_output_shapes	
:А*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp░
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/5100*
_output_shapes	
:А2+
)batch_normalization_5/AssignMovingAvg/subз
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/5100*
_output_shapes	
:А2+
)batch_normalization_5/AssignMovingAvg/mulБ
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_5_assignmovingavg_5100-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/5100*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_5/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/5106*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_5/AssignMovingAvg_1/decay┌
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_1_5106*
_output_shapes	
:А*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/5106*
_output_shapes	
:А2-
+batch_normalization_5/AssignMovingAvg_1/sub▒
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/5106*
_output_shapes	
:А2-
+batch_normalization_5/AssignMovingAvg_1/mulН
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_1_5106/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/5106*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/y█
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/addж
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_5/batchnorm/Rsqrtс
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▐
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/mul╪
%batch_normalization_5/batchnorm/mul_1Mul conv1d_8/conv1d/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_5/batchnorm/mul_1╘
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_5/batchnorm/mul_2╒
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp┌
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_5/batchnorm/subу
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*-
_output_shapes
:         АА2'
%batch_normalization_5/batchnorm/add_1В
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim║
max_pooling1d_2/ExpandDims
ExpandDimsadd_1/add:z:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         А─2
max_pooling1d_2/ExpandDims╧
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*0
_output_shapes
:         @─*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPoolн
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*,
_output_shapes
:         @─*
squeeze_dims
2
max_pooling1d_2/SqueezeС
activation_5/ReluRelu)batch_normalization_5/batchnorm/add_1:z:0*
T0*-
_output_shapes
:         АА2
activation_5/ReluЛ
conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_9/conv1d/ExpandDims/dim╠
conv1d_9/conv1d/ExpandDims
ExpandDimsactivation_5/Relu:activations:0'conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:         АА2
conv1d_9/conv1d/ExpandDims╒
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_9_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02-
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_9/conv1d/ExpandDims_1/dim▌
conv1d_9/conv1d/ExpandDims_1
ExpandDims3conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d_9/conv1d/ExpandDims_1█
conv1d_9/conv1dConv2D#conv1d_9/conv1d/ExpandDims:output:0%conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1d_9/conv1dо
conv1d_9/conv1d/SqueezeSqueezeconv1d_9/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d_9/conv1d/SqueezeЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_7/conv1d/ExpandDims/dim╠
conv1d_7/conv1d/ExpandDims
ExpandDims max_pooling1d_2/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @─2
conv1d_7/conv1d/ExpandDims╒
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:─А*
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim▌
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:─А2
conv1d_7/conv1d/ExpandDims_1█
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @А*
paddingSAME*
strides
2
conv1d_7/conv1dо
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:         @А*
squeeze_dims

¤        2
conv1d_7/conv1d/SqueezeЪ
	add_2/addAddV2 conv1d_9/conv1d/Squeeze:output:0 conv1d_7/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         @А2
	add_2/add╜
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_6/moments/mean/reduction_indices▌
"batch_normalization_6/moments/meanMeanadd_2/add:z:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2$
"batch_normalization_6/moments/mean├
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*#
_output_shapes
:А2,
*batch_normalization_6/moments/StopGradientЄ
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03batch_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:         @А21
/batch_normalization_6/moments/SquaredDifference┼
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_6/moments/variance/reduction_indicesП
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2(
&batch_normalization_6/moments/variance─
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze╠
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1▐
+batch_normalization_6/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/5154*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_6/AssignMovingAvg/decay╘
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_6_assignmovingavg_5154*
_output_shapes	
:А*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp░
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/5154*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/subз
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/5154*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/mulБ
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_6_assignmovingavg_5154-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_6/AssignMovingAvg/5154*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_6/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/5160*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_6/AssignMovingAvg_1/decay┌
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_1_5160*
_output_shapes	
:А*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/5160*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/sub▒
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/5160*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/mulН
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_1_5160/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg_1/5160*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/y█
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/addж
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtс
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▐
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mul─
%batch_normalization_6/batchnorm/mul_1Muladd_2/add:z:0'batch_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @А2'
%batch_normalization_6/batchnorm/mul_1╘
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2╒
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp┌
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/subт
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @А2'
%batch_normalization_6/batchnorm/add_1Р
activation_6/ReluRelu)batch_normalization_6/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @А2
activation_6/ReluН
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_11/conv1d/ExpandDims/dim╬
conv1d_11/conv1d/ExpandDims
ExpandDimsactivation_6/Relu:activations:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
conv1d_11/conv1d/ExpandDims╪
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimс
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d_11/conv1d/ExpandDims_1▀
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         @└*
paddingSAME*
strides
2
conv1d_11/conv1d▒
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*,
_output_shapes
:         @└*
squeeze_dims

¤        2
conv1d_11/conv1d/Squeeze╜
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indicesё
"batch_normalization_7/moments/meanMean!conv1d_11/conv1d/Squeeze:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2$
"batch_normalization_7/moments/mean├
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*#
_output_shapes
:└2,
*batch_normalization_7/moments/StopGradientЖ
/batch_normalization_7/moments/SquaredDifferenceSquaredDifference!conv1d_11/conv1d/Squeeze:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:         @└21
/batch_normalization_7/moments/SquaredDifference┼
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_7/moments/variance/reduction_indicesП
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2(
&batch_normalization_7/moments/variance─
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze╠
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1▐
+batch_normalization_7/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/5195*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_7/AssignMovingAvg/decay╘
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_7_assignmovingavg_5195*
_output_shapes	
:└*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp░
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/5195*
_output_shapes	
:└2+
)batch_normalization_7/AssignMovingAvg/subз
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/5195*
_output_shapes	
:└2+
)batch_normalization_7/AssignMovingAvg/mulБ
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_7_assignmovingavg_5195-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_7/AssignMovingAvg/5195*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_7/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/5201*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_7/AssignMovingAvg_1/decay┌
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_1_5201*
_output_shapes	
:└*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/5201*
_output_shapes	
:└2-
+batch_normalization_7/AssignMovingAvg_1/sub▒
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/5201*
_output_shapes	
:└2-
+batch_normalization_7/AssignMovingAvg_1/mulН
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_1_5201/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg_1/5201*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/y█
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/addж
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_7/batchnorm/Rsqrtс
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▐
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/mul╪
%batch_normalization_7/batchnorm/mul_1Mul!conv1d_11/conv1d/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2'
%batch_normalization_7/batchnorm/mul_1╘
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_7/batchnorm/mul_2╒
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp┌
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization_7/batchnorm/subт
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2'
%batch_normalization_7/batchnorm/add_1В
max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_3/ExpandDims/dim╣
max_pooling1d_3/ExpandDims
ExpandDimsadd_2/add:z:0'max_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @А2
max_pooling1d_3/ExpandDims╧
max_pooling1d_3/MaxPoolMaxPool#max_pooling1d_3/ExpandDims:output:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling1d_3/MaxPoolн
max_pooling1d_3/SqueezeSqueeze max_pooling1d_3/MaxPool:output:0*
T0*,
_output_shapes
:         А*
squeeze_dims
2
max_pooling1d_3/SqueezeР
activation_7/ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         @└2
activation_7/ReluН
conv1d_12/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_12/conv1d/ExpandDims/dim╬
conv1d_12/conv1d/ExpandDims
ExpandDimsactivation_7/Relu:activations:0(conv1d_12/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         @└2
conv1d_12/conv1d/ExpandDims╪
,conv1d_12/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_12_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:└└*
dtype02.
,conv1d_12/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_12/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_12/conv1d/ExpandDims_1/dimс
conv1d_12/conv1d/ExpandDims_1
ExpandDims4conv1d_12/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_12/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:└└2
conv1d_12/conv1d/ExpandDims_1▀
conv1d_12/conv1dConv2D$conv1d_12/conv1d/ExpandDims:output:0&conv1d_12/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1d_12/conv1d▒
conv1d_12/conv1d/SqueezeSqueezeconv1d_12/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d_12/conv1d/SqueezeН
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2!
conv1d_10/conv1d/ExpandDims/dim╧
conv1d_10/conv1d/ExpandDims
ExpandDims max_pooling1d_3/Squeeze:output:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d_10/conv1d/ExpandDims╪
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:А└*
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpИ
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimс
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:А└2
conv1d_10/conv1d/ExpandDims_1▀
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         └*
paddingSAME*
strides
2
conv1d_10/conv1d▒
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*,
_output_shapes
:         └*
squeeze_dims

¤        2
conv1d_10/conv1d/SqueezeЬ
	add_3/addAddV2!conv1d_12/conv1d/Squeeze:output:0!conv1d_10/conv1d/Squeeze:output:0*
T0*,
_output_shapes
:         └2
	add_3/add╜
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_8/moments/mean/reduction_indices▌
"batch_normalization_8/moments/meanMeanadd_3/add:z:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2$
"batch_normalization_8/moments/mean├
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*#
_output_shapes
:└2,
*batch_normalization_8/moments/StopGradientЄ
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03batch_normalization_8/moments/StopGradient:output:0*
T0*,
_output_shapes
:         └21
/batch_normalization_8/moments/SquaredDifference┼
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_8/moments/variance/reduction_indicesП
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:└*
	keep_dims(2(
&batch_normalization_8/moments/variance─
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze╠
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1▐
+batch_normalization_8/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/5249*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_8/AssignMovingAvg/decay╘
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_8_assignmovingavg_5249*
_output_shapes	
:└*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp░
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/5249*
_output_shapes	
:└2+
)batch_normalization_8/AssignMovingAvg/subз
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/5249*
_output_shapes	
:└2+
)batch_normalization_8/AssignMovingAvg/mulБ
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_8_assignmovingavg_5249-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_8/AssignMovingAvg/5249*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpф
-batch_normalization_8/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/5255*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_8/AssignMovingAvg_1/decay┌
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_8_assignmovingavg_1_5255*
_output_shapes	
:└*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp║
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/5255*
_output_shapes	
:└2-
+batch_normalization_8/AssignMovingAvg_1/sub▒
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/5255*
_output_shapes	
:└2-
+batch_normalization_8/AssignMovingAvg_1/mulН
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_8_assignmovingavg_1_5255/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg_1/5255*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_8/batchnorm/add/y█
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/addж
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_8/batchnorm/Rsqrtс
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOp▐
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/mul─
%batch_normalization_8/batchnorm/mul_1Muladd_3/add:z:0'batch_normalization_8/batchnorm/mul:z:0*
T0*,
_output_shapes
:         └2'
%batch_normalization_8/batchnorm/mul_1╘
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2'
%batch_normalization_8/batchnorm/mul_2╒
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOp┌
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization_8/batchnorm/subт
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*,
_output_shapes
:         └2'
%batch_normalization_8/batchnorm/add_1Р
activation_8/ReluRelu)batch_normalization_8/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         └2
activation_8/Relu~
embed/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
embed/Mean/reduction_indicesЫ

embed/MeanMeanactivation_8/Relu:activations:0%embed/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         └2

embed/Meanо	
IdentityIdentityembed/Mean:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*ё
_input_shapes▀
▄:         А :::::::::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
╘)
─
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6014

inputs
assignmovingavg_5989
assignmovingavg_1_5995)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradientк
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:         А А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/5989*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5989*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/5989*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/5989*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5989AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/5989*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/5995*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5995*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5995*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5995*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5995AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/5995*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:         А А2
batchnorm/add_1╗
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:         А А2

Identity"
identityIdentity:output:0*<
_input_shapes+
):         А А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:         А А
 
_user_specified_nameinputs
╨
m
'__inference_conv1d_4_layer_call_fn_6607

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         А─*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_4_layer_call_and_return_conditional_losses_33762
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
Е*
─
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7154

inputs
assignmovingavg_7129
assignmovingavg_1_7135)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ь
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/7129*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_7129*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp┬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/7129*
_output_shapes	
:А2
AssignMovingAvg/sub╣
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/7129*
_output_shapes	
:А2
AssignMovingAvg/mul¤
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_7129AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/7129*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/7135*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_7135*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╠
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7135*
_output_shapes	
:А2
AssignMovingAvg_1/sub├
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/7135*
_output_shapes	
:А2
AssignMovingAvg_1/mulЙ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_7135AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/7135*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1├
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Ш
Р
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2866

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А @2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А @:::::T P
,
_output_shapes
:         А @
 
_user_specified_nameinputs
еh
╩
__inference__traced_save_7819
file_prefix,
(savev2_conv1d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv1d_6_kernel_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv1d_8_kernel_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv1d_9_kernel_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop/
+savev2_conv1d_11_kernel_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_conv1d_12_kernel_read_readvariableop/
+savev2_conv1d_10_kernel_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e18f30f0f3084110ad0861f0e7de8c6b/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╫
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*щ
value▀B▄2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesь
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЗ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv1d_6_kernel_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv1d_8_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv1d_9_kernel_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop+savev2_conv1d_11_kernel_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_conv1d_12_kernel_read_readvariableop+savev2_conv1d_10_kernel_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
4222
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*▌
_input_shapes╦
╚: :@:@:@:@:@:@А:А:А:А:А:АА:@А:А:А:А:А:А─:─:─:─:─:──:А─:─:─:─:─:─А:А:А:А:А:АА:─А:А:А:А:А:А└:└:└:└:└:└└:А└:└:└:└:└: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!	

_output_shapes	
:А:!


_output_shapes	
:А:*&
$
_output_shapes
:АА:)%
#
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:*&
$
_output_shapes
:А─:!

_output_shapes	
:─:!

_output_shapes	
:─:!

_output_shapes	
:─:!

_output_shapes	
:─:*&
$
_output_shapes
:──:*&
$
_output_shapes
:А─:!

_output_shapes	
:─:!

_output_shapes	
:─:!

_output_shapes	
:─:!

_output_shapes	
:─:*&
$
_output_shapes
:─А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:! 

_output_shapes	
:А:*!&
$
_output_shapes
:АА:*"&
$
_output_shapes
:─А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:!%

_output_shapes	
:А:!&

_output_shapes	
:А:*'&
$
_output_shapes
:А└:!(

_output_shapes	
:└:!)

_output_shapes	
:└:!*

_output_shapes	
:└:!+

_output_shapes	
:└:*,&
$
_output_shapes
:└└:*-&
$
_output_shapes
:А└:!.

_output_shapes	
:└:!/

_output_shapes	
:└:!0

_output_shapes	
:└:!1

_output_shapes	
:└:2

_output_shapes
: 
В
Т
B__inference_conv1d_1_layer_call_and_return_conditional_losses_3080

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А@2
conv1d/ExpandDims╣
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@А*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╕
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@А2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:         АА*
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:         АА*
squeeze_dims

¤        2
conv1d/Squeezeq
IdentityIdentityconv1d/Squeeze:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::T P
,
_output_shapes
:         А@
 
_user_specified_nameinputs
╔
[
?__inference_embed_layer_call_and_return_conditional_losses_4104

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         └2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*+
_input_shapes
:         └:T P
,
_output_shapes
:         └
 
_user_specified_nameinputs
г
Т
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3882

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         @└2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         @└2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         @└:::::T P
,
_output_shapes
:         @└
 
_user_specified_nameinputs
╬
m
'__inference_conv1d_1_layer_call_fn_6190

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_30802
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@:22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А@
 
_user_specified_nameinputs
╠
b
F__inference_activation_5_layer_call_and_return_conditional_losses_6981

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:         АА2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs
щ
з
4__inference_batch_normalization_1_layer_call_fn_6129

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16802
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:                  А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╨
m
'__inference_conv1d_8_layer_call_fn_6812

inputs
unknown
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_8_layer_call_and_return_conditional_losses_35192
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А─:22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         А─
 
_user_specified_nameinputs
╔
k
?__inference_add_1_layer_call_and_return_conditional_losses_6613
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:         А─2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:         А─2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:         А─:         А─:W S
-
_output_shapes
:         А─
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         А─
"
_user_specified_name
inputs/1
√
Р
@__inference_conv1d_layer_call_and_return_conditional_losses_2799

inputs/
+conv1d_expanddims_1_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А 2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         А @*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         А @*
squeeze_dims

¤        2
conv1d/Squeezep
IdentityIdentityconv1d/Squeeze:output:0*
T0*,
_output_shapes
:         А @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А ::T P
,
_output_shapes
:         А 
 
_user_specified_nameinputs
о
G
+__inference_activation_2_layer_call_fn_6376

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_32032
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         АА2

Identity"
identityIdentity:output:0*,
_input_shapes
:         АА:U Q
-
_output_shapes
:         АА
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ж
serving_defaultТ
8
ecg1
serving_default_ecg:0         А :
embed1
StatefulPartitionedCall:0         └tensorflow/serving/predict:И╥	
ў┴
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer_with_weights-16
layer-29
layer-30
 layer_with_weights-17
 layer-31
!layer_with_weights-18
!layer-32
"layer-33
#layer-34
$layer_with_weights-19
$layer-35
%layer_with_weights-20
%layer-36
&layer-37
'layer_with_weights-21
'layer-38
(layer-39
)layer-40
*trainable_variables
+regularization_losses
,	variables
-	keras_api
.
signatures
╓_default_save_signature
+╫&call_and_return_all_conditional_losses
╪__call__"╗╖
_tf_keras_networkЮ╖{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "ecg"}, "name": "ecg", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["ecg", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["conv1d_3", 0, 0, {}], ["conv1d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv1d_5", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["conv1d_6", 0, 0, {}], ["conv1d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_8", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv1d_8", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_9", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["conv1d_9", 0, 0, {}], ["conv1d_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_11", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv1d_11", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_3", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_12", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_12", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_10", "inbound_nodes": [[["max_pooling1d_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["conv1d_12", 0, 0, {}], ["conv1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "embed", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "embed", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}], "input_layers": [["ecg", 0, 0]], "output_layers": [["embed", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "ecg"}, "name": "ecg", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["ecg", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["max_pooling1d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["conv1d_3", 0, 0, {}], ["conv1d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv1d_5", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["max_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["conv1d_6", 0, 0, {}], ["conv1d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_8", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv1d_8", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_9", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["conv1d_9", 0, 0, {}], ["conv1d_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_11", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv1d_11", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_3", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_12", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_12", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_10", "inbound_nodes": [[["max_pooling1d_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["conv1d_12", 0, 0, {}], ["conv1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "embed", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "embed", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}], "input_layers": [["ecg", 0, 0]], "output_layers": [["embed", 0, 0]]}}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "ecg", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4096, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "ecg"}}
╪	

/kernel
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"╗
_tf_keras_layerб{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 1]}}
╢	
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9trainable_variables
:regularization_losses
;	variables
<	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"р
_tf_keras_layer╞{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 64]}}
╙
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+▌&call_and_return_all_conditional_losses
▐__call__"┬
_tf_keras_layerи{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
▀	

Akernel
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+▀&call_and_return_all_conditional_losses
р__call__"┬
_tf_keras_layerи{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 64]}}
╝	
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+с&call_and_return_all_conditional_losses
т__call__"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 128]}}
╫
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ў
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"х
_tf_keras_layer╦{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
с	

Wkernel
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"─
_tf_keras_layerк{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096, 128]}}
▐	

\kernel
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"┴
_tf_keras_layerз{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 64]}}
╡
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"д
_tf_keras_layerК{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1024, 128]}, {"class_name": "TensorShape", "items": [null, 1024, 128]}]}
╝	
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 128]}}
╫
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
с	

rkernel
strainable_variables
tregularization_losses
u	variables
v	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"─
_tf_keras_layerк{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 128]}}
╝	
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 196}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 196]}}
█
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
■
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
+ў&call_and_return_all_conditional_losses
°__call__"щ
_tf_keras_layer╧{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц	
Иkernel
Йtrainable_variables
Кregularization_losses
Л	variables
М	keras_api
+∙&call_and_return_all_conditional_losses
·__call__"─
_tf_keras_layerк{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 196}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 196]}}
ф	
Нkernel
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
+√&call_and_return_all_conditional_losses
№__call__"┬
_tf_keras_layerи{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 196, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 128]}}
╗
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
+¤&call_and_return_all_conditional_losses
■__call__"ж
_tf_keras_layerМ{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256, 196]}, {"class_name": "TensorShape", "items": [null, 256, 196]}]}
─	
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ыtrainable_variables
Ьregularization_losses
Э	variables
Ю	keras_api
+ &call_and_return_all_conditional_losses
А__call__"х
_tf_keras_layer╦{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 196}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 196]}}
█
Яtrainable_variables
аregularization_losses
б	variables
в	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
х	
гkernel
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"├
_tf_keras_layerй{"class_name": "Conv1D", "name": "conv1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 196}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 196]}}
─	
	иaxis

йgamma
	кbeta
лmoving_mean
мmoving_variance
нtrainable_variables
оregularization_losses
п	variables
░	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"х
_tf_keras_layer╦{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256]}}
█
▒trainable_variables
▓regularization_losses
│	variables
┤	keras_api
+З&call_and_return_all_conditional_losses
И__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
■
╡trainable_variables
╢regularization_losses
╖	variables
╕	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"щ
_tf_keras_layer╧{"class_name": "MaxPooling1D", "name": "max_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
х	
╣kernel
║trainable_variables
╗regularization_losses
╝	variables
╜	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"├
_tf_keras_layerй{"class_name": "Conv1D", "name": "conv1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256]}}
у	
╛kernel
┐trainable_variables
└regularization_losses
┴	variables
┬	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"┴
_tf_keras_layerз{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 196}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 196]}}
╣
├trainable_variables
─regularization_losses
┼	variables
╞	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"д
_tf_keras_layerК{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 256]}, {"class_name": "TensorShape", "items": [null, 64, 256]}]}
├	
	╟axis

╚gamma
	╔beta
╩moving_mean
╦moving_variance
╠trainable_variables
═regularization_losses
╬	variables
╧	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 256]}}
█
╨trainable_variables
╤regularization_losses
╥	variables
╙	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
ц	
╘kernel
╒trainable_variables
╓regularization_losses
╫	variables
╪	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"─
_tf_keras_layerк{"class_name": "Conv1D", "name": "conv1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 256]}}
├	
	┘axis

┌gamma
	█beta
▄moving_mean
▌moving_variance
▐trainable_variables
▀regularization_losses
р	variables
с	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 320]}}
█
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
■
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"щ
_tf_keras_layer╧{"class_name": "MaxPooling1D", "name": "max_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц	
ъkernel
ыtrainable_variables
ьregularization_losses
э	variables
ю	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"─
_tf_keras_layerк{"class_name": "Conv1D", "name": "conv1d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_12", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 320]}}
х	
яkernel
Ёtrainable_variables
ёregularization_losses
Є	variables
є	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"├
_tf_keras_layerй{"class_name": "Conv1D", "name": "conv1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 320, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 256]}}
╣
Їtrainable_variables
їregularization_losses
Ў	variables
ў	keras_api
+б&call_and_return_all_conditional_losses
в__call__"д
_tf_keras_layerК{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 320]}, {"class_name": "TensorShape", "items": [null, 16, 320]}]}
├	
	°axis

∙gamma
	·beta
√moving_mean
№moving_variance
¤trainable_variables
■regularization_losses
 	variables
А	keras_api
+г&call_and_return_all_conditional_losses
д__call__"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 320]}}
█
Бtrainable_variables
Вregularization_losses
Г	variables
Д	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "relu"}}
є
Еtrainable_variables
Жregularization_losses
З	variables
И	keras_api
+з&call_and_return_all_conditional_losses
и__call__"▐
_tf_keras_layer─{"class_name": "GlobalAveragePooling1D", "name": "embed", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "embed", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
а
/0
51
62
A3
G4
H5
W6
\7
f8
g9
r10
x11
y12
И13
Н14
Ч15
Ш16
г17
й18
к19
╣20
╛21
╚22
╔23
╘24
┌25
█26
ъ27
я28
∙29
·30"
trackable_list_wrapper
 "
trackable_list_wrapper
║
/0
51
62
73
84
A5
G6
H7
I8
J9
W10
\11
f12
g13
h14
i15
r16
x17
y18
z19
{20
И21
Н22
Ч23
Ш24
Щ25
Ъ26
г27
й28
к29
л30
м31
╣32
╛33
╚34
╔35
╩36
╦37
╘38
┌39
█40
▄41
▌42
ъ43
я44
∙45
·46
√47
№48"
trackable_list_wrapper
╙
*trainable_variables
+regularization_losses
 Йlayer_regularization_losses
,	variables
Кmetrics
Лlayer_metrics
Мlayers
Нnon_trainable_variables
╪__call__
╓_default_save_signature
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
-
йserving_default"
signature_map
#:!@2conv1d/kernel
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
╡
0trainable_variables
 Оlayer_regularization_losses
1regularization_losses
2	variables
Пmetrics
Рlayer_metrics
Сlayers
Тnon_trainable_variables
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
╡
9trainable_variables
 Уlayer_regularization_losses
:regularization_losses
;	variables
Фmetrics
Хlayer_metrics
Цlayers
Чnon_trainable_variables
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
=trainable_variables
 Шlayer_regularization_losses
>regularization_losses
?	variables
Щmetrics
Ъlayer_metrics
Ыlayers
Ьnon_trainable_variables
▐__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
&:$@А2conv1d_2/kernel
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
╡
Btrainable_variables
 Эlayer_regularization_losses
Cregularization_losses
D	variables
Юmetrics
Яlayer_metrics
аlayers
бnon_trainable_variables
р__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_1/gamma
):'А2batch_normalization_1/beta
2:0А (2!batch_normalization_1/moving_mean
6:4А (2%batch_normalization_1/moving_variance
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
G0
H1
I2
J3"
trackable_list_wrapper
╡
Ktrainable_variables
 вlayer_regularization_losses
Lregularization_losses
M	variables
гmetrics
дlayer_metrics
еlayers
жnon_trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Otrainable_variables
 зlayer_regularization_losses
Pregularization_losses
Q	variables
иmetrics
йlayer_metrics
кlayers
лnon_trainable_variables
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Strainable_variables
 мlayer_regularization_losses
Tregularization_losses
U	variables
нmetrics
оlayer_metrics
пlayers
░non_trainable_variables
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
':%АА2conv1d_3/kernel
'
W0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
W0"
trackable_list_wrapper
╡
Xtrainable_variables
 ▒layer_regularization_losses
Yregularization_losses
Z	variables
▓metrics
│layer_metrics
┤layers
╡non_trainable_variables
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
&:$@А2conv1d_1/kernel
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
\0"
trackable_list_wrapper
╡
]trainable_variables
 ╢layer_regularization_losses
^regularization_losses
_	variables
╖metrics
╕layer_metrics
╣layers
║non_trainable_variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
atrainable_variables
 ╗layer_regularization_losses
bregularization_losses
c	variables
╝metrics
╜layer_metrics
╛layers
┐non_trainable_variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_2/gamma
):'А2batch_normalization_2/beta
2:0А (2!batch_normalization_2/moving_mean
6:4А (2%batch_normalization_2/moving_variance
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
╡
jtrainable_variables
 └layer_regularization_losses
kregularization_losses
l	variables
┴metrics
┬layer_metrics
├layers
─non_trainable_variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
ntrainable_variables
 ┼layer_regularization_losses
oregularization_losses
p	variables
╞metrics
╟layer_metrics
╚layers
╔non_trainable_variables
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
':%А─2conv1d_5/kernel
'
r0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
r0"
trackable_list_wrapper
╡
strainable_variables
 ╩layer_regularization_losses
tregularization_losses
u	variables
╦metrics
╠layer_metrics
═layers
╬non_trainable_variables
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(─2batch_normalization_3/gamma
):'─2batch_normalization_3/beta
2:0─ (2!batch_normalization_3/moving_mean
6:4─ (2%batch_normalization_3/moving_variance
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
╡
|trainable_variables
 ╧layer_regularization_losses
}regularization_losses
~	variables
╨metrics
╤layer_metrics
╥layers
╙non_trainable_variables
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Аtrainable_variables
 ╘layer_regularization_losses
Бregularization_losses
В	variables
╒metrics
╓layer_metrics
╫layers
╪non_trainable_variables
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Дtrainable_variables
 ┘layer_regularization_losses
Еregularization_losses
Ж	variables
┌metrics
█layer_metrics
▄layers
▌non_trainable_variables
°__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
':%──2conv1d_6/kernel
(
И0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
И0"
trackable_list_wrapper
╕
Йtrainable_variables
 ▐layer_regularization_losses
Кregularization_losses
Л	variables
▀metrics
рlayer_metrics
сlayers
тnon_trainable_variables
·__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
':%А─2conv1d_4/kernel
(
Н0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
╕
Оtrainable_variables
 уlayer_regularization_losses
Пregularization_losses
Р	variables
фmetrics
хlayer_metrics
цlayers
чnon_trainable_variables
№__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Тtrainable_variables
 шlayer_regularization_losses
Уregularization_losses
Ф	variables
щmetrics
ъlayer_metrics
ыlayers
ьnon_trainable_variables
■__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(─2batch_normalization_4/gamma
):'─2batch_normalization_4/beta
2:0─ (2!batch_normalization_4/moving_mean
6:4─ (2%batch_normalization_4/moving_variance
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
╕
Ыtrainable_variables
 эlayer_regularization_losses
Ьregularization_losses
Э	variables
юmetrics
яlayer_metrics
Ёlayers
ёnon_trainable_variables
А__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Яtrainable_variables
 Єlayer_regularization_losses
аregularization_losses
б	variables
єmetrics
Їlayer_metrics
їlayers
Ўnon_trainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
':%─А2conv1d_8/kernel
(
г0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
г0"
trackable_list_wrapper
╕
дtrainable_variables
 ўlayer_regularization_losses
еregularization_losses
ж	variables
°metrics
∙layer_metrics
·layers
√non_trainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_5/gamma
):'А2batch_normalization_5/beta
2:0А (2!batch_normalization_5/moving_mean
6:4А (2%batch_normalization_5/moving_variance
0
й0
к1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
й0
к1
л2
м3"
trackable_list_wrapper
╕
нtrainable_variables
 №layer_regularization_losses
оregularization_losses
п	variables
¤metrics
■layer_metrics
 layers
Аnon_trainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▒trainable_variables
 Бlayer_regularization_losses
▓regularization_losses
│	variables
Вmetrics
Гlayer_metrics
Дlayers
Еnon_trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╡trainable_variables
 Жlayer_regularization_losses
╢regularization_losses
╖	variables
Зmetrics
Иlayer_metrics
Йlayers
Кnon_trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
':%АА2conv1d_9/kernel
(
╣0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
╣0"
trackable_list_wrapper
╕
║trainable_variables
 Лlayer_regularization_losses
╗regularization_losses
╝	variables
Мmetrics
Нlayer_metrics
Оlayers
Пnon_trainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
':%─А2conv1d_7/kernel
(
╛0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
╛0"
trackable_list_wrapper
╕
┐trainable_variables
 Рlayer_regularization_losses
└regularization_losses
┴	variables
Сmetrics
Тlayer_metrics
Уlayers
Фnon_trainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
├trainable_variables
 Хlayer_regularization_losses
─regularization_losses
┼	variables
Цmetrics
Чlayer_metrics
Шlayers
Щnon_trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_6/gamma
):'А2batch_normalization_6/beta
2:0А (2!batch_normalization_6/moving_mean
6:4А (2%batch_normalization_6/moving_variance
0
╚0
╔1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
╚0
╔1
╩2
╦3"
trackable_list_wrapper
╕
╠trainable_variables
 Ъlayer_regularization_losses
═regularization_losses
╬	variables
Ыmetrics
Ьlayer_metrics
Эlayers
Юnon_trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╨trainable_variables
 Яlayer_regularization_losses
╤regularization_losses
╥	variables
аmetrics
бlayer_metrics
вlayers
гnon_trainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
(:&А└2conv1d_11/kernel
(
╘0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
╘0"
trackable_list_wrapper
╕
╒trainable_variables
 дlayer_regularization_losses
╓regularization_losses
╫	variables
еmetrics
жlayer_metrics
зlayers
иnon_trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(└2batch_normalization_7/gamma
):'└2batch_normalization_7/beta
2:0└ (2!batch_normalization_7/moving_mean
6:4└ (2%batch_normalization_7/moving_variance
0
┌0
█1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
┌0
█1
▄2
▌3"
trackable_list_wrapper
╕
▐trainable_variables
 йlayer_regularization_losses
▀regularization_losses
р	variables
кmetrics
лlayer_metrics
мlayers
нnon_trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
тtrainable_variables
 оlayer_regularization_losses
уregularization_losses
ф	variables
пmetrics
░layer_metrics
▒layers
▓non_trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
цtrainable_variables
 │layer_regularization_losses
чregularization_losses
ш	variables
┤metrics
╡layer_metrics
╢layers
╖non_trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
(:&└└2conv1d_12/kernel
(
ъ0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
ъ0"
trackable_list_wrapper
╕
ыtrainable_variables
 ╕layer_regularization_losses
ьregularization_losses
э	variables
╣metrics
║layer_metrics
╗layers
╝non_trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
(:&А└2conv1d_10/kernel
(
я0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
я0"
trackable_list_wrapper
╕
Ёtrainable_variables
 ╜layer_regularization_losses
ёregularization_losses
Є	variables
╛metrics
┐layer_metrics
└layers
┴non_trainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Їtrainable_variables
 ┬layer_regularization_losses
їregularization_losses
Ў	variables
├metrics
─layer_metrics
┼layers
╞non_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(└2batch_normalization_8/gamma
):'└2batch_normalization_8/beta
2:0└ (2!batch_normalization_8/moving_mean
6:4└ (2%batch_normalization_8/moving_variance
0
∙0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
∙0
·1
√2
№3"
trackable_list_wrapper
╕
¤trainable_variables
 ╟layer_regularization_losses
■regularization_losses
 	variables
╚metrics
╔layer_metrics
╩layers
╦non_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Бtrainable_variables
 ╠layer_regularization_losses
Вregularization_losses
Г	variables
═metrics
╬layer_metrics
╧layers
╨non_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Еtrainable_variables
 ╤layer_regularization_losses
Жregularization_losses
З	variables
╥metrics
╙layer_metrics
╘layers
╒non_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40"
trackable_list_wrapper
░
70
81
I2
J3
h4
i5
z6
{7
Щ8
Ъ9
л10
м11
╩12
╦13
▄14
▌15
√16
№17"
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
.
70
81"
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
.
I0
J1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
h0
i1"
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
.
z0
{1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Щ0
Ъ1"
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
0
л0
м1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
╩0
╦1"
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
0
▄0
▌1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
√0
№1"
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
▐2█
__inference__wrapped_model_1444╖
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *'в$
"К
ecg         А 
ц2у
F__inference_functional_1_layer_call_and_return_conditional_losses_5277
F__inference_functional_1_layer_call_and_return_conditional_losses_4112
F__inference_functional_1_layer_call_and_return_conditional_losses_5560
F__inference_functional_1_layer_call_and_return_conditional_losses_4254└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·2ў
+__inference_functional_1_layer_call_fn_4500
+__inference_functional_1_layer_call_fn_4745
+__inference_functional_1_layer_call_fn_5766
+__inference_functional_1_layer_call_fn_5663└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
@__inference_conv1d_layer_call_and_return_conditional_losses_5778в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_conv1d_layer_call_fn_5785в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5841
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5923
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5903
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5821┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
2__inference_batch_normalization_layer_call_fn_5949
2__inference_batch_normalization_layer_call_fn_5854
2__inference_batch_normalization_layer_call_fn_5867
2__inference_batch_normalization_layer_call_fn_5936┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
D__inference_activation_layer_call_and_return_conditional_losses_5954в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_activation_layer_call_fn_5959в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_2_layer_call_and_return_conditional_losses_5971в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_2_layer_call_fn_5978в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6014
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6096
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6034
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6116┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_1_layer_call_fn_6129
4__inference_batch_normalization_1_layer_call_fn_6047
4__inference_batch_normalization_1_layer_call_fn_6142
4__inference_batch_normalization_1_layer_call_fn_6060┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_1_layer_call_and_return_conditional_losses_6147в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_1_layer_call_fn_6152в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
в2Я
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_1733╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
З2Д
,__inference_max_pooling1d_layer_call_fn_1739╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
ь2щ
B__inference_conv1d_3_layer_call_and_return_conditional_losses_6164в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_3_layer_call_fn_6171в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_1_layer_call_and_return_conditional_losses_6183в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_1_layer_call_fn_6190в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_add_layer_call_and_return_conditional_losses_6196в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
"__inference_add_layer_call_fn_6202в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6238
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6258
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6320
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6340┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_2_layer_call_fn_6353
4__inference_batch_normalization_2_layer_call_fn_6366
4__inference_batch_normalization_2_layer_call_fn_6271
4__inference_batch_normalization_2_layer_call_fn_6284┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_2_layer_call_and_return_conditional_losses_6371в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_2_layer_call_fn_6376в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_5_layer_call_and_return_conditional_losses_6388в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_5_layer_call_fn_6395в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6431
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6451
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6513
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6533┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_3_layer_call_fn_6464
4__inference_batch_normalization_3_layer_call_fn_6477
4__inference_batch_normalization_3_layer_call_fn_6546
4__inference_batch_normalization_3_layer_call_fn_6559┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_3_layer_call_and_return_conditional_losses_6564в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_3_layer_call_fn_6569в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
д2б
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2028╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
Й2Ж
.__inference_max_pooling1d_1_layer_call_fn_2034╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
ь2щ
B__inference_conv1d_6_layer_call_and_return_conditional_losses_6581в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_6_layer_call_fn_6588в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_4_layer_call_and_return_conditional_losses_6600в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_4_layer_call_fn_6607в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_add_1_layer_call_and_return_conditional_losses_6613в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_add_1_layer_call_fn_6619в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6737
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6655
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6675
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6757┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_4_layer_call_fn_6783
4__inference_batch_normalization_4_layer_call_fn_6770
4__inference_batch_normalization_4_layer_call_fn_6701
4__inference_batch_normalization_4_layer_call_fn_6688┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_4_layer_call_and_return_conditional_losses_6788в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_4_layer_call_fn_6793в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_8_layer_call_and_return_conditional_losses_6805в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_8_layer_call_fn_6812в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6868
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6950
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6848
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6930┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_5_layer_call_fn_6881
4__inference_batch_normalization_5_layer_call_fn_6976
4__inference_batch_normalization_5_layer_call_fn_6894
4__inference_batch_normalization_5_layer_call_fn_6963┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_5_layer_call_and_return_conditional_losses_6981в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_5_layer_call_fn_6986в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
д2б
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2323╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
Й2Ж
.__inference_max_pooling1d_2_layer_call_fn_2329╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
ь2щ
B__inference_conv1d_9_layer_call_and_return_conditional_losses_6998в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_9_layer_call_fn_7005в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv1d_7_layer_call_and_return_conditional_losses_7017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv1d_7_layer_call_fn_7024в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_add_2_layer_call_and_return_conditional_losses_7030в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_add_2_layer_call_fn_7036в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7154
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7092
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7174
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7072┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_6_layer_call_fn_7187
4__inference_batch_normalization_6_layer_call_fn_7200
4__inference_batch_normalization_6_layer_call_fn_7118
4__inference_batch_normalization_6_layer_call_fn_7105┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_6_layer_call_and_return_conditional_losses_7205в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_6_layer_call_fn_7210в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv1d_11_layer_call_and_return_conditional_losses_7222в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_conv1d_11_layer_call_fn_7229в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7285
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7265
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7347
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7367┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_7_layer_call_fn_7298
4__inference_batch_normalization_7_layer_call_fn_7380
4__inference_batch_normalization_7_layer_call_fn_7393
4__inference_batch_normalization_7_layer_call_fn_7311┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_7_layer_call_and_return_conditional_losses_7398в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_7_layer_call_fn_7403в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
д2б
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2618╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
Й2Ж
.__inference_max_pooling1d_3_layer_call_fn_2624╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
э2ъ
C__inference_conv1d_12_layer_call_and_return_conditional_losses_7415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_conv1d_12_layer_call_fn_7422в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv1d_10_layer_call_and_return_conditional_losses_7434в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_conv1d_10_layer_call_fn_7441в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_add_3_layer_call_and_return_conditional_losses_7447в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_add_3_layer_call_fn_7453в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7571
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7509
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7489
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7591┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
4__inference_batch_normalization_8_layer_call_fn_7604
4__inference_batch_normalization_8_layer_call_fn_7535
4__inference_batch_normalization_8_layer_call_fn_7522
4__inference_batch_normalization_8_layer_call_fn_7617┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
F__inference_activation_8_layer_call_and_return_conditional_losses_7622в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_8_layer_call_fn_7627в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╖2┤
?__inference_embed_layer_call_and_return_conditional_losses_7644
?__inference_embed_layer_call_and_return_conditional_losses_7633п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Б2■
$__inference_embed_layer_call_fn_7649
$__inference_embed_layer_call_fn_7638п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-B+
"__inference_signature_wrapper_4850ecg╓
__inference__wrapped_model_1444▓M/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·1в.
'в$
"К
ecg         А 
к ".к+
)
embed К
embed         └о
F__inference_activation_1_layer_call_and_return_conditional_losses_6147d5в2
+в(
&К#
inputs         А А
к "+в(
!К
0         А А
Ъ Ж
+__inference_activation_1_layer_call_fn_6152W5в2
+в(
&К#
inputs         А А
к "К         А Ао
F__inference_activation_2_layer_call_and_return_conditional_losses_6371d5в2
+в(
&К#
inputs         АА
к "+в(
!К
0         АА
Ъ Ж
+__inference_activation_2_layer_call_fn_6376W5в2
+в(
&К#
inputs         АА
к "К         ААо
F__inference_activation_3_layer_call_and_return_conditional_losses_6564d5в2
+в(
&К#
inputs         А─
к "+в(
!К
0         А─
Ъ Ж
+__inference_activation_3_layer_call_fn_6569W5в2
+в(
&К#
inputs         А─
к "К         А─о
F__inference_activation_4_layer_call_and_return_conditional_losses_6788d5в2
+в(
&К#
inputs         А─
к "+в(
!К
0         А─
Ъ Ж
+__inference_activation_4_layer_call_fn_6793W5в2
+в(
&К#
inputs         А─
к "К         А─о
F__inference_activation_5_layer_call_and_return_conditional_losses_6981d5в2
+в(
&К#
inputs         АА
к "+в(
!К
0         АА
Ъ Ж
+__inference_activation_5_layer_call_fn_6986W5в2
+в(
&К#
inputs         АА
к "К         ААм
F__inference_activation_6_layer_call_and_return_conditional_losses_7205b4в1
*в'
%К"
inputs         @А
к "*в'
 К
0         @А
Ъ Д
+__inference_activation_6_layer_call_fn_7210U4в1
*в'
%К"
inputs         @А
к "К         @Ам
F__inference_activation_7_layer_call_and_return_conditional_losses_7398b4в1
*в'
%К"
inputs         @└
к "*в'
 К
0         @└
Ъ Д
+__inference_activation_7_layer_call_fn_7403U4в1
*в'
%К"
inputs         @└
к "К         @└м
F__inference_activation_8_layer_call_and_return_conditional_losses_7622b4в1
*в'
%К"
inputs         └
к "*в'
 К
0         └
Ъ Д
+__inference_activation_8_layer_call_fn_7627U4в1
*в'
%К"
inputs         └
к "К         └к
D__inference_activation_layer_call_and_return_conditional_losses_5954b4в1
*в'
%К"
inputs         А @
к "*в'
 К
0         А @
Ъ В
)__inference_activation_layer_call_fn_5959U4в1
*в'
%К"
inputs         А @
к "К         А @┘
?__inference_add_1_layer_call_and_return_conditional_losses_6613Хfвc
\вY
WЪT
(К%
inputs/0         А─
(К%
inputs/1         А─
к "+в(
!К
0         А─
Ъ ▒
$__inference_add_1_layer_call_fn_6619Иfвc
\вY
WЪT
(К%
inputs/0         А─
(К%
inputs/1         А─
к "К         А─╓
?__inference_add_2_layer_call_and_return_conditional_losses_7030Тdвa
ZвW
UЪR
'К$
inputs/0         @А
'К$
inputs/1         @А
к "*в'
 К
0         @А
Ъ о
$__inference_add_2_layer_call_fn_7036Еdвa
ZвW
UЪR
'К$
inputs/0         @А
'К$
inputs/1         @А
к "К         @А╓
?__inference_add_3_layer_call_and_return_conditional_losses_7447Тdвa
ZвW
UЪR
'К$
inputs/0         └
'К$
inputs/1         └
к "*в'
 К
0         └
Ъ о
$__inference_add_3_layer_call_fn_7453Еdвa
ZвW
UЪR
'К$
inputs/0         └
'К$
inputs/1         └
к "К         └╫
=__inference_add_layer_call_and_return_conditional_losses_6196Хfвc
\вY
WЪT
(К%
inputs/0         АА
(К%
inputs/1         АА
к "+в(
!К
0         АА
Ъ п
"__inference_add_layer_call_fn_6202Иfвc
\вY
WЪT
(К%
inputs/0         АА
(К%
inputs/1         АА
к "К         АА┴
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6014nIJGH9в6
/в,
&К#
inputs         А А
p
к "+в(
!К
0         А А
Ъ ┴
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6034nJGIH9в6
/в,
&К#
inputs         А А
p 
к "+в(
!К
0         А А
Ъ ╤
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6096~IJGHAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ ╤
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6116~JGIHAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ Щ
4__inference_batch_normalization_1_layer_call_fn_6047aIJGH9в6
/в,
&К#
inputs         А А
p
к "К         А АЩ
4__inference_batch_normalization_1_layer_call_fn_6060aJGIH9в6
/в,
&К#
inputs         А А
p 
к "К         А Ай
4__inference_batch_normalization_1_layer_call_fn_6129qIJGHAв>
7в4
.К+
inputs                  А
p
к "&К#                  Ай
4__inference_batch_normalization_1_layer_call_fn_6142qJGIHAв>
7в4
.К+
inputs                  А
p 
к "&К#                  А┴
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6238nhifg9в6
/в,
&К#
inputs         АА
p
к "+в(
!К
0         АА
Ъ ┴
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6258nifhg9в6
/в,
&К#
inputs         АА
p 
к "+в(
!К
0         АА
Ъ ╤
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6320~hifgAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ ╤
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6340~ifhgAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ Щ
4__inference_batch_normalization_2_layer_call_fn_6271ahifg9в6
/в,
&К#
inputs         АА
p
к "К         ААЩ
4__inference_batch_normalization_2_layer_call_fn_6284aifhg9в6
/в,
&К#
inputs         АА
p 
к "К         ААй
4__inference_batch_normalization_2_layer_call_fn_6353qhifgAв>
7в4
.К+
inputs                  А
p
к "&К#                  Ай
4__inference_batch_normalization_2_layer_call_fn_6366qifhgAв>
7в4
.К+
inputs                  А
p 
к "&К#                  А╤
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6431~z{xyAв>
7в4
.К+
inputs                  ─
p
к "3в0
)К&
0                  ─
Ъ ╤
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6451~{xzyAв>
7в4
.К+
inputs                  ─
p 
к "3в0
)К&
0                  ─
Ъ ┴
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6513nz{xy9в6
/в,
&К#
inputs         А─
p
к "+в(
!К
0         А─
Ъ ┴
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6533n{xzy9в6
/в,
&К#
inputs         А─
p 
к "+в(
!К
0         А─
Ъ й
4__inference_batch_normalization_3_layer_call_fn_6464qz{xyAв>
7в4
.К+
inputs                  ─
p
к "&К#                  ─й
4__inference_batch_normalization_3_layer_call_fn_6477q{xzyAв>
7в4
.К+
inputs                  ─
p 
к "&К#                  ─Щ
4__inference_batch_normalization_3_layer_call_fn_6546az{xy9в6
/в,
&К#
inputs         А─
p
к "К         А─Щ
4__inference_batch_normalization_3_layer_call_fn_6559a{xzy9в6
/в,
&К#
inputs         А─
p 
к "К         А─╓
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6655ВЩЪЧШAв>
7в4
.К+
inputs                  ─
p
к "3в0
)К&
0                  ─
Ъ ╓
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6675ВЪЧЩШAв>
7в4
.К+
inputs                  ─
p 
к "3в0
)К&
0                  ─
Ъ ┼
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6737rЩЪЧШ9в6
/в,
&К#
inputs         А─
p
к "+в(
!К
0         А─
Ъ ┼
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6757rЪЧЩШ9в6
/в,
&К#
inputs         А─
p 
к "+в(
!К
0         А─
Ъ н
4__inference_batch_normalization_4_layer_call_fn_6688uЩЪЧШAв>
7в4
.К+
inputs                  ─
p
к "&К#                  ─н
4__inference_batch_normalization_4_layer_call_fn_6701uЪЧЩШAв>
7в4
.К+
inputs                  ─
p 
к "&К#                  ─Э
4__inference_batch_normalization_4_layer_call_fn_6770eЩЪЧШ9в6
/в,
&К#
inputs         А─
p
к "К         А─Э
4__inference_batch_normalization_4_layer_call_fn_6783eЪЧЩШ9в6
/в,
&К#
inputs         А─
p 
к "К         А─┼
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6848rлмйк9в6
/в,
&К#
inputs         АА
p
к "+в(
!К
0         АА
Ъ ┼
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6868rмйлк9в6
/в,
&К#
inputs         АА
p 
к "+в(
!К
0         АА
Ъ ╓
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6930ВлмйкAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ ╓
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6950ВмйлкAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ Э
4__inference_batch_normalization_5_layer_call_fn_6881eлмйк9в6
/в,
&К#
inputs         АА
p
к "К         ААЭ
4__inference_batch_normalization_5_layer_call_fn_6894eмйлк9в6
/в,
&К#
inputs         АА
p 
к "К         ААн
4__inference_batch_normalization_5_layer_call_fn_6963uлмйкAв>
7в4
.К+
inputs                  А
p
к "&К#                  Ан
4__inference_batch_normalization_5_layer_call_fn_6976uмйлкAв>
7в4
.К+
inputs                  А
p 
к "&К#                  А├
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7072p╩╦╚╔8в5
.в+
%К"
inputs         @А
p
к "*в'
 К
0         @А
Ъ ├
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7092p╦╚╩╔8в5
.в+
%К"
inputs         @А
p 
к "*в'
 К
0         @А
Ъ ╓
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7154В╩╦╚╔Aв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ ╓
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7174В╦╚╩╔Aв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ Ы
4__inference_batch_normalization_6_layer_call_fn_7105c╩╦╚╔8в5
.в+
%К"
inputs         @А
p
к "К         @АЫ
4__inference_batch_normalization_6_layer_call_fn_7118c╦╚╩╔8в5
.в+
%К"
inputs         @А
p 
к "К         @Ан
4__inference_batch_normalization_6_layer_call_fn_7187u╩╦╚╔Aв>
7в4
.К+
inputs                  А
p
к "&К#                  Ан
4__inference_batch_normalization_6_layer_call_fn_7200u╦╚╩╔Aв>
7в4
.К+
inputs                  А
p 
к "&К#                  А├
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7265p▄▌┌█8в5
.в+
%К"
inputs         @└
p
к "*в'
 К
0         @└
Ъ ├
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7285p▌┌▄█8в5
.в+
%К"
inputs         @└
p 
к "*в'
 К
0         @└
Ъ ╓
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7347В▄▌┌█Aв>
7в4
.К+
inputs                  └
p
к "3в0
)К&
0                  └
Ъ ╓
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7367В▌┌▄█Aв>
7в4
.К+
inputs                  └
p 
к "3в0
)К&
0                  └
Ъ Ы
4__inference_batch_normalization_7_layer_call_fn_7298c▄▌┌█8в5
.в+
%К"
inputs         @└
p
к "К         @└Ы
4__inference_batch_normalization_7_layer_call_fn_7311c▌┌▄█8в5
.в+
%К"
inputs         @└
p 
к "К         @└н
4__inference_batch_normalization_7_layer_call_fn_7380u▄▌┌█Aв>
7в4
.К+
inputs                  └
p
к "&К#                  └н
4__inference_batch_normalization_7_layer_call_fn_7393u▌┌▄█Aв>
7в4
.К+
inputs                  └
p 
к "&К#                  └├
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7489p√№∙·8в5
.в+
%К"
inputs         └
p
к "*в'
 К
0         └
Ъ ├
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7509p№∙√·8в5
.в+
%К"
inputs         └
p 
к "*в'
 К
0         └
Ъ ╓
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7571В√№∙·Aв>
7в4
.К+
inputs                  └
p
к "3в0
)К&
0                  └
Ъ ╓
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_7591В№∙√·Aв>
7в4
.К+
inputs                  └
p 
к "3в0
)К&
0                  └
Ъ Ы
4__inference_batch_normalization_8_layer_call_fn_7522c√№∙·8в5
.в+
%К"
inputs         └
p
к "К         └Ы
4__inference_batch_normalization_8_layer_call_fn_7535c№∙√·8в5
.в+
%К"
inputs         └
p 
к "К         └н
4__inference_batch_normalization_8_layer_call_fn_7604u√№∙·Aв>
7в4
.К+
inputs                  └
p
к "&К#                  └н
4__inference_batch_normalization_8_layer_call_fn_7617u№∙√·Aв>
7в4
.К+
inputs                  └
p 
к "&К#                  └╜
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5821l78568в5
.в+
%К"
inputs         А @
p
к "*в'
 К
0         А @
Ъ ╜
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5841l85768в5
.в+
%К"
inputs         А @
p 
к "*в'
 К
0         А @
Ъ ═
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5903|7856@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ═
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5923|8576@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ Х
2__inference_batch_normalization_layer_call_fn_5854_78568в5
.в+
%К"
inputs         А @
p
к "К         А @Х
2__inference_batch_normalization_layer_call_fn_5867_85768в5
.в+
%К"
inputs         А @
p 
к "К         А @е
2__inference_batch_normalization_layer_call_fn_5936o7856@в=
6в3
-К*
inputs                  @
p
к "%К"                  @е
2__inference_batch_normalization_layer_call_fn_5949o8576@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @н
C__inference_conv1d_10_layer_call_and_return_conditional_losses_7434fя4в1
*в'
%К"
inputs         А
к "*в'
 К
0         └
Ъ Е
(__inference_conv1d_10_layer_call_fn_7441Yя4в1
*в'
%К"
inputs         А
к "К         └н
C__inference_conv1d_11_layer_call_and_return_conditional_losses_7222f╘4в1
*в'
%К"
inputs         @А
к "*в'
 К
0         @└
Ъ Е
(__inference_conv1d_11_layer_call_fn_7229Y╘4в1
*в'
%К"
inputs         @А
к "К         @└н
C__inference_conv1d_12_layer_call_and_return_conditional_losses_7415fъ4в1
*в'
%К"
inputs         @└
к "*в'
 К
0         └
Ъ Е
(__inference_conv1d_12_layer_call_fn_7422Yъ4в1
*в'
%К"
inputs         @└
к "К         └м
B__inference_conv1d_1_layer_call_and_return_conditional_losses_6183f\4в1
*в'
%К"
inputs         А@
к "+в(
!К
0         АА
Ъ Д
'__inference_conv1d_1_layer_call_fn_6190Y\4в1
*в'
%К"
inputs         А@
к "К         ААм
B__inference_conv1d_2_layer_call_and_return_conditional_losses_5971fA4в1
*в'
%К"
inputs         А @
к "+в(
!К
0         А А
Ъ Д
'__inference_conv1d_2_layer_call_fn_5978YA4в1
*в'
%К"
inputs         А @
к "К         А Ан
B__inference_conv1d_3_layer_call_and_return_conditional_losses_6164gW5в2
+в(
&К#
inputs         А А
к "+в(
!К
0         АА
Ъ Е
'__inference_conv1d_3_layer_call_fn_6171ZW5в2
+в(
&К#
inputs         А А
к "К         ААо
B__inference_conv1d_4_layer_call_and_return_conditional_losses_6600hН5в2
+в(
&К#
inputs         АА
к "+в(
!К
0         А─
Ъ Ж
'__inference_conv1d_4_layer_call_fn_6607[Н5в2
+в(
&К#
inputs         АА
к "К         А─н
B__inference_conv1d_5_layer_call_and_return_conditional_losses_6388gr5в2
+в(
&К#
inputs         АА
к "+в(
!К
0         А─
Ъ Е
'__inference_conv1d_5_layer_call_fn_6395Zr5в2
+в(
&К#
inputs         АА
к "К         А─о
B__inference_conv1d_6_layer_call_and_return_conditional_losses_6581hИ5в2
+в(
&К#
inputs         А─
к "+в(
!К
0         А─
Ъ Ж
'__inference_conv1d_6_layer_call_fn_6588[И5в2
+в(
&К#
inputs         А─
к "К         А─м
B__inference_conv1d_7_layer_call_and_return_conditional_losses_7017f╛4в1
*в'
%К"
inputs         @─
к "*в'
 К
0         @А
Ъ Д
'__inference_conv1d_7_layer_call_fn_7024Y╛4в1
*в'
%К"
inputs         @─
к "К         @Ао
B__inference_conv1d_8_layer_call_and_return_conditional_losses_6805hг5в2
+в(
&К#
inputs         А─
к "+в(
!К
0         АА
Ъ Ж
'__inference_conv1d_8_layer_call_fn_6812[г5в2
+в(
&К#
inputs         А─
к "К         ААн
B__inference_conv1d_9_layer_call_and_return_conditional_losses_6998g╣5в2
+в(
&К#
inputs         АА
к "*в'
 К
0         @А
Ъ Е
'__inference_conv1d_9_layer_call_fn_7005Z╣5в2
+в(
&К#
inputs         АА
к "К         @Ай
@__inference_conv1d_layer_call_and_return_conditional_losses_5778e/4в1
*в'
%К"
inputs         А 
к "*в'
 К
0         А @
Ъ Б
%__inference_conv1d_layer_call_fn_5785X/4в1
*в'
%К"
inputs         А 
к "К         А @е
?__inference_embed_layer_call_and_return_conditional_losses_7633b8в5
.в+
%К"
inputs         └

 
к "&в#
К
0         └
Ъ ╛
?__inference_embed_layer_call_and_return_conditional_losses_7644{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ }
$__inference_embed_layer_call_fn_7638U8в5
.в+
%К"
inputs         └

 
к "К         └Ц
$__inference_embed_layer_call_fn_7649nIвF
?в<
6К3
inputs'                           

 
к "!К                  ¤
F__inference_functional_1_layer_call_and_return_conditional_losses_4112▓M/7856AIJGHW\hifgrz{xyИНЩЪЧШглмйк╣╛╩╦╚╔╘▄▌┌█ъя√№∙·9в6
/в,
"К
ecg         А 
p

 
к "&в#
К
0         └
Ъ ¤
F__inference_functional_1_layer_call_and_return_conditional_losses_4254▓M/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·9в6
/в,
"К
ecg         А 
p 

 
к "&в#
К
0         └
Ъ А
F__inference_functional_1_layer_call_and_return_conditional_losses_5277╡M/7856AIJGHW\hifgrz{xyИНЩЪЧШглмйк╣╛╩╦╚╔╘▄▌┌█ъя√№∙·<в9
2в/
%К"
inputs         А 
p

 
к "&в#
К
0         └
Ъ А
F__inference_functional_1_layer_call_and_return_conditional_losses_5560╡M/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·<в9
2в/
%К"
inputs         А 
p 

 
к "&в#
К
0         └
Ъ ╒
+__inference_functional_1_layer_call_fn_4500еM/7856AIJGHW\hifgrz{xyИНЩЪЧШглмйк╣╛╩╦╚╔╘▄▌┌█ъя√№∙·9в6
/в,
"К
ecg         А 
p

 
к "К         └╒
+__inference_functional_1_layer_call_fn_4745еM/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·9в6
/в,
"К
ecg         А 
p 

 
к "К         └╪
+__inference_functional_1_layer_call_fn_5663иM/7856AIJGHW\hifgrz{xyИНЩЪЧШглмйк╣╛╩╦╚╔╘▄▌┌█ъя√№∙·<в9
2в/
%К"
inputs         А 
p

 
к "К         └╪
+__inference_functional_1_layer_call_fn_5766иM/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·<в9
2в/
%К"
inputs         А 
p 

 
к "К         └╥
I__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2028ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ й
.__inference_max_pooling1d_1_layer_call_fn_2034wEвB
;в8
6К3
inputs'                           
к ".К+'                           ╥
I__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2323ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ й
.__inference_max_pooling1d_2_layer_call_fn_2329wEвB
;в8
6К3
inputs'                           
к ".К+'                           ╥
I__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2618ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ й
.__inference_max_pooling1d_3_layer_call_fn_2624wEвB
;в8
6К3
inputs'                           
к ".К+'                           ╨
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_1733ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ з
,__inference_max_pooling1d_layer_call_fn_1739wEвB
;в8
6К3
inputs'                           
к ".К+'                           р
"__inference_signature_wrapper_4850╣M/8576AJGIHW\ifhgr{xzyИНЪЧЩШгмйлк╣╛╦╚╩╔╘▌┌▄█ъя№∙√·8в5
в 
.к+
)
ecg"К
ecg         А ".к+
)
embed К
embed         └