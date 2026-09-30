## Cada dataset tem três colunas:

Setpoint, ControllerOutput, PlantOutput

## Nos poisoned datasets, apenas o PlantOutput foi afetado comparativamente ao original.

## Como a janela do regime transitório foi definida como sendo 100 samples, o regime
## transitório é definido como sendo o primeiro valor após o setpoint ir de 0 -> 1 ou
## de 1 -> 0 mais 99 samples.

### Exemplo: 

Setpoint, ControllerOutput, PlantOutput
0         _                 W
0         _                 X
1         _                 Y
1         _                 Z

### O início do regime transitório é o primeiro setpoint "1", que corresponde a planta "Y".

## Importante destacar que, se um dataset começa com o setpoint = 1, isso conta como o
## início duma janela de regime transitório