#include <math.h>
/******************************************************************
 * Network Configuration
 ******************************************************************/
const int PatternCount =1;
const int OutputNodes = 1;
const int InputNodes = 3;
const float pi=3.1416;
float X_K=0;
float X_K_1=0;
float X_K_2=0;
float D;
float alfa=0.1;
float OutputWeights[OutputNodes][InputNodes+1]= {
   {-0.5, 0.5,1.0,-0.5}
   }; 
/******************************************************************
 * End Network Configuration
 ******************************************************************/
int i, j;
float Accum;
float Output[OutputNodes];
float Input[InputNodes][PatternCount];
 
void setup(){
  //start serial connection
  Serial.begin(9600);
}
void loop(){
  float Entrada;
  float Salida;
  float Tiempo;
  float SenoOri;
  float SenoRuido;
Entrada=millis();
Tiempo=Entrada/1000; 
SenoOri=sin(2*pi*0.05*Tiempo);
SenoRuido=sin(2*pi*0.05*Tiempo)+0.50*sin(2*pi*0.050*10*Tiempo);
X_K=SenoRuido;
Input[0][0]=X_K;
Input[1][0]=X_K_1;
Input[2][0]=X_K_2;
D=SenoOri;
/******************************************************************
* Compute output layer activations
******************************************************************/
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[i][InputNodes] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += OutputWeights[i][j]*Input[j][0];
      }
      Output[i] = Accum; 
    }
Salida=Output[0];
OutputWeights[0][0]=OutputWeights[0][0]+alfa*(D-Salida)*X_K;
OutputWeights[0][1]=OutputWeights[0][1]+alfa*(D-Salida)*X_K_1;
OutputWeights[0][2]=OutputWeights[0][2]+alfa*(D-Salida)*X_K_2;
OutputWeights[0][3]=OutputWeights[0][3]+alfa*(D-Salida);
X_K_2=X_K_1;
X_K_1=X_K;

   Serial.print(Salida);       // print as an ASCII-encoded decimal - same as "DEC"
   Serial.print("\t");    // prints a tab
   Serial.print(SenoRuido);       // print as an ASCII-encoded decimal - same as "DEC"
   Serial.print("\t");    // prints a tab
   Serial.println(SenoOri);       // print as an ASCII-encoded decimal - same as "DEC"
   
delay(10);  
}



