# Plug and play project

## Problèmes Inverses et Méthodes Plug-and-Play

### Problème Inverse

Les problèmes inverses sont des défis courants en traitement d'images où l'objectif est de reconstruire une image originale à partir de ses observations dégradées. Par exemple, il s'agit de récupérer une photo nette à partir d'une version floue ou de restaurer des parties manquantes d'une image.

Mathématiquement, cela peut être exprimé par :

$$
\mathbf{y} = \mathbf{A} \mathbf{x} + \mathbf{n},
$$

où :

- $\mathbf{y}$ est l'image observée, souvent dégradée.
- $\mathbf{A}$ représente l'opération qui a dégradé l'image originale (comme le flou, la réduction de résolution, etc.).
- $\mathbf{x}$ est l'image originale que nous cherchons à reconstruire.
- $\mathbf{n}$ est le bruit ajouté lors de l'observation.

### Formulation du Problème

Pour résoudre un problème inverse, nous voulons trouver une image $\mathbf{x}$ qui soit compatible avec les observations $\mathbf{y}$ tout en respectant certaines propriétés désirées (comme la netteté ou la régularité). Cela peut être formulé comme un problème d'optimisation :

$$
\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{y} - \mathbf{A} \mathbf{x}\|^2_2 + \lambda R(\mathbf{x}),
$$

où :

- $\frac{1}{2} \|\mathbf{y} - \mathbf{A} \mathbf{x}\|^2_2$ est le terme de fidélité aux données, assurant que notre estimation $\mathbf{x}$ est proche de l'observation $\mathbf{y}$.
- $R(\mathbf{x})$ est un terme de régularisation qui impose des contraintes supplémentaires sur $\mathbf{x}$ pour obtenir une solution réaliste.
- $\lambda$ est un paramètre qui équilibre l'importance de la fidélité aux données et de la régularisation.

### Méthodes Plug-and-Play

Les méthodes Plug-and-Play (PnP) sont une solution flexible et puissante pour résoudre les problèmes inverses. Plutôt que de définir explicitement le terme de régularisation $R(\mathbf{x})$, les méthodes PnP utilisent des débruiteurs avancés comme régularisateurs implicites. Ces débruiteurs peuvent être des réseaux de neurones préentraînés ou d'autres modèles sophistiqués.

#### Utilisation de l'Algorithme ADMM

L'algorithme ADMM (Alternating Direction Method of Multipliers) est souvent utilisé avec les méthodes PnP pour résoudre des problèmes inverses. Voici comment cela fonctionne :

1. **Formulation du problème avec une variable auxiliaire** :
   On reformule le problème d'optimisation en introduisant une variable auxiliaire $\mathbf{z}$ :
   $$\min_{\mathbf{x}, \mathbf{z}} \frac{1}{2} \|\mathbf{y} - \mathbf{A} \mathbf{x}\|^2_2 + \lambda R(\mathbf{z}) \quad \text{sous les contraintes} \quad \mathbf{x} = \mathbf{z}.$$

2. **Mise à jour de $\mathbf{x}$** :
   La mise à jour de $\mathbf{x}$ se fait en minimisant un terme qui combine la fidélité aux données et la proximité de $\mathbf{x}$ à $\mathbf{z}$ de l'itération précédente :
   $$\mathbf{x}_{k+1} = \arg \min_{\mathbf{x}} \left( \frac{1}{2} \|\mathbf{y} - \mathbf{A} \mathbf{x}\|^2_2 + \frac{\rho}{2} \|\mathbf{x} - \mathbf{z}_k + \frac{1}{\rho} \mathbf{u}_k \|^2_2 \right),$$
   où $\mathbf{u}_k$ est une variable duale mise à jour à chaque itération.

3. **Mise à jour de $\mathbf{z}$** :
   La mise à jour de $\mathbf{z}$ utilise le débruiteur Plug-and-Play pour imposer la régularisation :
   $$\mathbf{z}_{k+1} = \arg \min_{\mathbf{z}} \left( \lambda R(\mathbf{z}) + \frac{\rho}{2} \|\mathbf{x}_{k+1} - \mathbf{z} + \frac{1}{\rho} \mathbf{u}_k \|^2_2 \right).$$

4. **Mise à jour de la variable duale $\mathbf{u}$** :
   Enfin, la variable duale est mise à jour pour ajuster la contrainte $\mathbf{x} = \mathbf{z}$ :
   $$\mathbf{u}_{k+1} = \mathbf{u}_k + \rho (\mathbf{x}_{k+1} - \mathbf{z}_{k+1}).$$

#### Utilisation de la Descente de Gradient

En plus de l'algorithme ADMM, la descente de gradient est une autre méthode populaire pour résoudre des problèmes inverses en utilisant des techniques Plug-and-Play. Voici comment cela peut être appliqué :

1. **Formulation du problème** :
   Comme pour ADMM, nous cherchons à minimiser une fonction de coût composée d'un terme de fidélité aux données et d'un terme de régularisation implicite.

2. **Algorithme de descente de gradient avec débruitage** :
   L'algorithme de descente de gradient est modifié pour intégrer un débruiteur dans chaque itération. La mise à jour de l'estimation $\mathbf{x}$ se fait en deux étapes :

   - **Mise à jour par descente de gradient** :
     $$\mathbf{x}_{k+1/2} = \mathbf{x}_k - \alpha \nabla_{\mathbf{x}} \left( \frac{1}{2} \|\mathbf{y} - \mathbf{A} \mathbf{x}_k\|^2_2 \right),$$
     où $\alpha$ est le pas de la descente de gradient et $\nabla_{\mathbf{x}}$ est le gradient de la fonction de coût par rapport à $\mathbf{x}$.

   - **Débruitage Plug-and-Play** :
     $$\mathbf{x}_{k+1} = D_{\sigma}(\mathbf{x}_{k+1/2}),$$
     où $D_{\sigma}$ est le débruiteur utilisé comme modèle de régularisation implicite.

3. **Itération** :
   Ces étapes sont répétées jusqu'à convergence, c'est-à-dire jusqu'à ce que la différence entre deux itérations successives soit suffisamment petite.

### Avantages des Méthodes Plug-and-Play

Les méthodes Plug-and-Play offrent plusieurs avantages importants pour la résolution des problèmes inverses :

1. **Flexibilité** : Elles permettent d'incorporer divers types de débruiteurs, y compris ceux basés sur des réseaux de neurones profonds, des autoencodeurs, ou des techniques par ondelettes.
2. **Efficacité** : En utilisant des débruiteurs préentraînés, les méthodes PnP peuvent bénéficier des avancées récentes en apprentissage automatique, conduisant à une reconstruction d'image de haute qualité.
3. **Modularité** : Elles séparent la régularisation de l'optimisation, ce qui permet de modifier ou de remplacer les débruiteurs sans changer l'algorithme d'optimisation sous-jacent.

En résumé, les méthodes Plug-and-Play représentent une approche puissante et flexible pour traiter les problèmes inverses en intégrant des modèles de priors sophistiqués dans le processus d'optimisation, ce qui améliore considérablement la qualité de la reconstruction d'images. Que ce soit par l'algorithme ADMM ou la descente de gradient, ces méthodes permettent d'utiliser des débruiteurs avancés pour obtenir des résultats optimaux.
