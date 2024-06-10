1. Comparison of Different methods

    Training GMF model...

   Epoch 1/10, Loss: 2.4708
   Epoch 2/10, Loss: 1.5142
   Epoch 3/10, Loss: 1.5264
   Epoch 4/10, Loss: 1.5240
   Epoch 5/10, Loss: 1.5204
   Epoch 6/10, Loss: 1.5174
   Epoch 7/10, Loss: 1.5151
   Epoch 8/10, Loss: 1.5133
   Epoch 9/10, Loss: 1.5117
   Epoch 10/10, Loss: 1.5090
   GMF: HR@10 = 0.0032, NDCG@10 = 0.9994

   Training MLP model...

   Epoch 1/10, Loss: 1.2955
   Epoch 2/10, Loss: 1.0943
   Epoch 3/10, Loss: 1.0472
   Epoch 4/10, Loss: 1.0340
   Epoch 5/10, Loss: 1.0319
   Epoch 6/10, Loss: 1.0323
   Epoch 7/10, Loss: 1.0364
   Epoch 8/10, Loss: 1.0366
   Epoch 9/10, Loss: 1.0324
   Epoch 10/10, Loss: 1.0289
   MLP: HR@10 = 0.0028, NDCG@10 = 0.9995

   Training NeuMF model...

   Epoch 1, Training Loss: 1.5296
   Epoch 2, Training Loss: 0.9486
   Epoch 3, Training Loss: 0.8664
   Epoch 4, Training Loss: 0.8374
   Epoch 5, Training Loss: 0.8244
   Epoch 6, Training Loss: 0.8174
   Epoch 7, Training Loss: 0.8120
   Epoch 8, Training Loss: 0.8068
   Epoch 9, Training Loss: 0.8021
   Epoch 10, Training Loss: 0.7970
   Test Loss: 0.8466
   HR@10: 0.0044, NDCG@10: 0.0084





2. Comparison of MLPs

   Training MLP with layers: [64]
   Epoch 1/10, Loss: 1.2500001496789703
   Epoch 2/10, Loss: 0.8638352488029941
   Epoch 3/10, Loss: 0.829849406228337
   Epoch 4/10, Loss: 0.8156330167713336
   Epoch 5/10, Loss: 0.806661166674955
   Epoch 6/10, Loss: 0.7996613234186203
   Epoch 7/10, Loss: 0.7931372552099551
   Epoch 8/10, Loss: 0.7878058413168748
   Epoch 9/10, Loss: 0.7822257720806319
   Epoch 10/10, Loss: 0.7770137269567086
   

   Training MLP with layers: [64, 32]
   Epoch 1/10, Loss: 1.1415147532352026
   Epoch 2/10, Loss: 0.8607760529753038
   Epoch 3/10, Loss: 0.8307911821343696
   Epoch 4/10, Loss: 0.817910741363972
   Epoch 5/10, Loss: 0.80878028597728
   Epoch 6/10, Loss: 0.8020904069708008
   Epoch 7/10, Loss: 0.7959040127651705
   Epoch 8/10, Loss: 0.7898605819016943
   Epoch 9/10, Loss: 0.7843578794943699
   Epoch 10/10, Loss: 0.7786161919015383


   Training MLP with layers: [64, 32, 16]
   Epoch 1/10, Loss: 1.1381639327403452
   Epoch 2/10, Loss: 0.8587942970386317
   Epoch 3/10, Loss: 0.8305190385760821
   Epoch 4/10, Loss: 0.8183395559979949
   Epoch 5/10, Loss: 0.8099842967128266
   Epoch 6/10, Loss: 0.803353118709624
   Epoch 7/10, Loss: 0.7971906647114745
   Epoch 8/10, Loss: 0.7916178422254854
   Epoch 9/10, Loss: 0.7854166742287915
   Epoch 10/10, Loss: 0.7799612237944942
   

   Training MLP with layers: [64]
   Epoch 1/10, Loss: 1.2621740356364162
   Epoch 2/10, Loss: 0.8661154565030157
   Epoch 3/10, Loss: 0.8306752778136876
   Epoch 4/10, Loss: 0.8160234923860009
   Epoch 5/10, Loss: 0.8066816197056383
   Epoch 6/10, Loss: 0.7992048274418214
   Epoch 7/10, Loss: 0.7929775508565164
   Epoch 8/10, Loss: 0.7872573693288265
   Epoch 9/10, Loss: 0.7815581949300211
   Epoch 10/10, Loss: 0.7764575096825644


   Training MLP with layers: [64, 32]
   Epoch 1/10, Loss: 1.1451951238640745
   Epoch 2/10, Loss: 0.8610479335943553
   Epoch 3/10, Loss: 0.8303593122188815
   Epoch 4/10, Loss: 0.8176549372769134
   Epoch 5/10, Loss: 0.8090439152046418
   Epoch 6/10, Loss: 0.8023324015616455
   Epoch 7/10, Loss: 0.7960227087797908
   Epoch 8/10, Loss: 0.7904597406233265
   Epoch 9/10, Loss: 0.7843121836106135
   Epoch 10/10, Loss: 0.7788187997202345


   Training MLP with layers: [64, 32, 16]
   Epoch 1/10, Loss: 1.2005845211632191
   Epoch 2/10, Loss: 0.8584618272494599
   Epoch 3/10, Loss: 0.8291799046065818
   Epoch 4/10, Loss: 0.8171285539960831
   Epoch 5/10, Loss: 0.8085296259274181
   Epoch 6/10, Loss: 0.8020635754994986
   Epoch 7/10, Loss: 0.7962136285585695
   Epoch 8/10, Loss: 0.7904831145104123
   Epoch 9/10, Loss: 0.7849347887897979
   Epoch 10/10, Loss: 0.7793965054221894


MLP with layers [64]:
HR@10 = 0.0253, NDCG@10 = 0.0198

MLP with layers [64, 32]:
HR@10 = 0.0274, NDCG@10 = 0.0216

MLP with layers [64, 32, 16]:
HR@10 = 0.0269, NDCG@10 = 0.0211

