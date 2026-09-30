+------------------+
|     MODELOS      |
+------------------+

Os modelos encontram-se na pasta **saved\_models** e estão organizados da seguinte forma:

saved\_models/
├── top25/
├── top50/
├── top75/
├── top100/
&#x20;   ├── Epsilon 0/                         # run\_X\_order\_2\_normal\_model.pth 
&#x20;   └── Epsilon {0.05, 0.1, 0.15, 0.2}/
&#x20;       ├── Calculated/                    # run\_X\_order\_2\_<Epsilon>\_calculated\_model.pth
&#x20;       └── Gaussian/                      # run\_X\_order\_2\_<Epsilon>\_gaussian\_model.pth
└── README.md


Onde X é o número da run: [0 - 29] (30 modelos) e <Epsilon> é o valor de Epsilon da pasta onde se encontra.

A nomenclatura topY indica o valor de τ utilizado. Por exemplo, top25 inclui todos os modelos onde a percentagem da perturbação é 25%.


+------------------+
|     DATASETS     |
+------------------+

Os modelos encontram-se na pasta **Datasets** e estão organizados da seguinte forma:

Datasets/
├── top25/
├── top50/
├── top75/
├── top100/
&#x20;   └── adversarialTop100_{0.05, 0.1, 0.15, 0.2}/
&#x20;       ├── Calculated/                    # top100\_calculated\_X.csv
&#x20;       └── Gaussian/                      # top100\_gaussian\_X.csv
├── Normal Datasets/                            # DCMotor_X.csv
└── README.md