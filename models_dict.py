import models

"""Dictionary of constructor calls to all models implemented for MIR project"""

models_dict = {
	"Conv_1_layer": models.Conv1Layer,
	"Conv_3_layer": models.Conv3Layer,
	"Conv_5_layer": models.Conv5Layer,
	"Conv_N_layer": models.ConvNLayer,
}