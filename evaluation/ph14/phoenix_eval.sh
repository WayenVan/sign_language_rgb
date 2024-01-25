#!/bin/bash

if [ -z "$5" ];then

echo "evaluateWER.sh <work_dir> <ctm> <stm> <prefix> <py_script>"
exit 0
fi

workdir=$1
hypothesisCTM=$2
groundtruthSTM=$3
tmp_prefix=$4
py_script=$5

cd $workdir
basenameCTM=`basename $hypothesisCTM`

# apply some simplifications to the recognition
cat ${hypothesisCTM} | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > $tmp_prefix.tmp.ctm

#make sure empty recognition results get filled with [EMPTY] tags - so that the alignment can work out on all data.
cat $tmp_prefix.tmp.ctm | sed -e 's,\s*$,,'   | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' |sort -k1,1 -k3,3 > $tmp_prefix.tmp2.ctm

#cat final_phoenix_noPause_noCompound_lefthandtag_noClean.stm  >${workdir}$tmp_prefix.tmp.stm

outSTM=$tmp_prefix.out.stm
cat $groundtruthSTM | sort  -k1,1 > $outSTM

#add missing entries, so that sclite can generate alignment
$py_script $tmp_prefix.tmp2.ctm $outSTM

outCTM=$tmp_prefix.out.$basenameCTM
mv $tmp_prefix.tmp2.ctm $outCTM

#make sure NIST sclite toolbox is installed and on path. Available at ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.0-20091110-0958.tar.bz2
sclite  -h $outCTM ctm -r $outSTM stm -f 0 -o sgml sum rsum pra
sclite  -h $outCTM ctm -r $outSTM stm -f 0 -o dtl stdout |grep Error

#cat ${workdir}$tmp_prefix.tmp.out.${hypothesisCTM}.raw
#cat ${workdir}$tmp_prefix.tmp.out.${hypothesisCTM}.sys
#rm ${workdir}$tmp_prefix.tmp.*
