#!/bin/bash


hypothesisCTM=$1

# apply some simplifications to the recognition
cat ${hypothesisCTM} | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' |
  perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'|
  perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| 
  perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| 
  perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > tmp.ctm 

#make sure empty recognition results get filled with [EMPTY] tags - so that the alignment can work out on all data.