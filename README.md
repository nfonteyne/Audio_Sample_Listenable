# Audio_Sample_Listenable

L'application "prediction" : 

L'application sert à tester des fichier audio unique sur les modèles déjà entrainés.

Pour l'utiliser, vous trouverez toutes les informations dans le rapport dédié qui se trouve dans le dossier "prediction_streamlit"

Si vous voulez mettre à jour l'application pour qu'elle fonctionne avec un modèle plus récent, vous n'avez qu'a modifier le dossier "my_model" en le remplacant par un dossier similaire.
Ces dossier peuvent être générés automatiquement par la commande "model.save()" qui est intégrée à la bibliothèque python "keras"

## Install dependencies with

```
$pip install -r requirements.txt
```

Mettre les fichier dont vous voulez tester l'écoutabilité dans le fichier "Sound_files"

éxécuter avec python le fichier Prediction.py

Les résultats devraient apparaître directement dans la console
