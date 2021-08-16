# CREDIT RISK ANALYSIS

## UNBALANCED OVERVIEW
Predicting models like credit risk often contain imbalanced data.  For example, When looking to predict bad loans, your data will contain much fewer items with bad loans than good.  When this happens, the data is called **Unbalanced**.  This can result in predictions skewed towards the larger population.

![oh no](images/confused.png)

## SOLUTION
We need to match the data sizes to optimize machine learning results.  We have two options:

1.  Add items to the short list ( Over Sampling)
2.  Remove items from the large list ( Under Sampling)

**Let's try them out and see what we discover...**

## METHODOLOGY

We will resample the unbalanced data four different ways and see if any are better than the others.

For the classification reports, 0 represents bad loans ( which we are trying to predict) and 1 represents good loans

### OVER SAMPLING EXAMPLES

#### Over Sampling Classification Report
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.025142</td>
      <td>0.998304</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.712644</td>
      <td>0.859563</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.859563</td>
      <td>0.712644</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.048570</td>
      <td>0.923753</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.782664</td>
      <td>0.782664</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.603562</td>
      <td>0.621562</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.993383</td>
      <td>0.85882</td>
      <td>0.713387</td>
      <td>0.919328</td>
      <td>0.782664</td>
      <td>0.621471</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>

#### SMOTE Over Sampling Classification Report
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.029287</td>
      <td>0.998343</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.712644</td>
      <td>0.879951</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.879951</td>
      <td>0.712644</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.056261</td>
      <td>0.935416</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.791891</td>
      <td>0.791891</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.616600</td>
      <td>0.637583</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.993443</td>
      <td>0.879105</td>
      <td>0.71349</td>
      <td>0.93097</td>
      <td>0.791891</td>
      <td>0.637477</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>


### UNDER SAMPLING EXAMPLES

#### Under Sampling Classification Report

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.020283</td>
      <td>0.998495</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.758621</td>
      <td>0.813763</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.813763</td>
      <td>0.758621</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.039509</td>
      <td>0.896714</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.785708</td>
      <td>0.785708</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.613934</td>
      <td>0.620742</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.993548</td>
      <td>0.813484</td>
      <td>0.7589</td>
      <td>0.892379</td>
      <td>0.785708</td>
      <td>0.620707</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>

#### SMOTEENN Combined Sampling Classification Report

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.016753</td>
      <td>0.998555</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.781609</td>
      <td>0.766854</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.766854</td>
      <td>0.781609</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.032803</td>
      <td>0.867499</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.774196</td>
      <td>0.774196</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.600264</td>
      <td>0.598495</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.99359</td>
      <td>0.766928</td>
      <td>0.781535</td>
      <td>0.863279</td>
      <td>0.774196</td>
      <td>0.598504</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>

## RESAMPLING CONCLUSION

Upon review of the above results, the SMOTE over sampling technique seems to produce the best results.  I will focus on 0 labeled outcomes ( bad loans) since we are trying to predict bad debt.  The precision of 0.029287 is the best and recall of 0.712644 is as good as the Random Over Sampling.  That recall is better than either under sample technique.  Therefore:
 **SMOTE Over Sampling seems the best technique to deal with this unbalanced data**

## PROBLEM

Even though resampling seems to provide some improvement, the results are still hardly useable or helpful for predictions.  

Another thing we can look at is ensemble learning models.   

 ## ENSEMBLE MACHINE LEARNING OVERVIEW

 Ensemble machine learning models combine smaller models to attempt to come up with better over all models.

 Now let's look at a couple ensemble models and see how they perform with this data.

### Balanced Random Forest Classifier

 <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.731707</td>
      <td>0.996679</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.344828</td>
      <td>0.999357</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.999357</td>
      <td>0.344828</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.468750</td>
      <td>0.998016</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.587032</td>
      <td>0.587032</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.322051</td>
      <td>0.367161</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.995339</td>
      <td>0.996048</td>
      <td>0.348137</td>
      <td>0.99534</td>
      <td>0.587032</td>
      <td>0.366933</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>


### Easy Ensemble Ada Boost Classifier

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>avg_pre</th>
      <th>avg_rec</th>
      <th>avg_spe</th>
      <th>avg_f1</th>
      <th>avg_geo</th>
      <th>avg_iba</th>
      <th>total_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pre</th>
      <td>0.074740</td>
      <td>0.999505</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>rec</th>
      <td>0.908046</td>
      <td>0.942867</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>spe</th>
      <td>0.942867</td>
      <td>0.908046</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.138112</td>
      <td>0.970360</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>geo</th>
      <td>0.925293</td>
      <td>0.925293</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>iba</th>
      <td>0.853185</td>
      <td>0.859148</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
    <tr>
      <th>sup</th>
      <td>87.000000</td>
      <td>17118.000000</td>
      <td>0.994828</td>
      <td>0.942691</td>
      <td>0.908222</td>
      <td>0.966152</td>
      <td>0.925293</td>
      <td>0.859118</td>
      <td>17205</td>
    </tr>
  </tbody>
</table>

## AND THE WINNER IS...

**Balanced Random Forest Classifier**

With a precision of .73 and recall of .34, this model is looking much better than the others.

![High Five](images/HighFive.png)
