# Plan — Intégration du provider OpenRouter

## Objectif

Ajouter OpenRouter comme cinquième provider de modèles dans Sinew. Contrairement aux quatre providers existants, il ne s'authentifie pas par OAuth mais par une clé API saisie directement dans la page Providers. Et surtout, il ne peuple pas automatiquement le sélecteur de modèles avec son catalogue : l'utilisateur compose lui-même la liste des modèles OpenRouter qu'il veut voir apparaître dans Sinew, en les recherchant et en les ajoutant un par un.

## Le bloc OpenRouter dans la page Providers

OpenRouter apparaît comme une nouvelle carte dans la section Providers, placée à la suite des cartes existantes (Anthropic, OpenAI, Google, Kimi). Là où les autres cartes exposent un bouton de connexion OAuth, la carte OpenRouter contient :

- Un en-tête identique aux autres cartes (logo OpenRouter, nom, description courte, badge d'état de connexion).
- Un champ de saisie pour la clé API OpenRouter, masqué par défaut, avec un bouton pour révéler temporairement la valeur.
- Sous le champ, une zone de recherche de modèles, qui n'est active qu'une fois qu'une clé valide a été enregistrée.
- Sous la recherche, la liste des modèles OpenRouter actuellement ajoutés par l'utilisateur, avec un bouton de suppression à côté de chacun.

## Saisie et validation de la clé API

Quand l'utilisateur colle ou tape sa clé API, l'application la valide automatiquement en arrière-plan (avec un léger délai pour éviter une validation à chaque caractère). La validation se fait en interrogeant un endpoint léger d'OpenRouter qui confirme que la clé est acceptée.

États visibles de la carte selon le résultat :

- Aucune clé saisie → état "Non connecté", la recherche est désactivée.
- Validation en cours → état "Connexion…" avec un indicateur de chargement.
- Clé valide → état "Connecté", la recherche devient active, et la clé est persistée.
- Clé invalide ou erreur réseau → état "Attention requise" avec un message d'erreur explicite, la clé n'est pas considérée comme valide.

La clé est stockée dans un fichier d'authentification dédié, dans le même répertoire et selon la même convention que les credentials des autres providers, afin de rester cohérent avec l'existant et hors de la base SQLite des préférences.

Un bouton "Déconnecter" permet d'effacer la clé à tout moment. Cela ne supprime pas la liste des modèles déjà ajoutés (pour permettre à l'utilisateur de la retrouver après une simple rotation de clé), mais ces modèles deviennent indisponibles dans les sélecteurs tant qu'aucune clé valide n'est présente.

## Recherche en direct dans le catalogue OpenRouter

Lorsque la clé est valide et que l'utilisateur tape dans la zone de recherche, l'application interroge OpenRouter à chaque frappe avec un debounce court pour éviter les requêtes inutiles. Les requêtes en cours sont annulées dès qu'une nouvelle frappe survient.

Les résultats sont affichés sous la zone de recherche, sous forme d'une liste compacte. Chaque ligne ne montre que le nom lisible du modèle. Un bouton "Ajouter" se trouve à droite de chaque résultat. Si le modèle est déjà présent dans la liste curatée de l'utilisateur, le bouton est remplacé par une indication "Déjà ajouté".

États de la zone de résultats :

- Pas de saisie → la zone est vide ou affiche un message d'invitation.
- Recherche en cours → indicateur de chargement discret.
- Résultats trouvés → liste tronquée raisonnablement (par exemple les vingt premiers).
- Aucun résultat → message neutre "Aucun modèle ne correspond".
- Erreur réseau ou clé révoquée pendant la recherche → message d'erreur lisible, la liste reste vide et le badge de la carte repasse en "Attention requise".

## Ajout d'un modèle à la liste curatée

Au moment où l'utilisateur clique sur "Ajouter" pour un résultat, l'application capture en une seule fois toutes les métadonnées utiles depuis la réponse d'OpenRouter, sans nécessiter d'autre interaction :

- Identifiant OpenRouter du modèle (au format provider sous-jacent / nom).
- Nom lisible affiché ensuite dans le sélecteur.
- Fenêtre de contexte maximale.
- Limite de tokens de sortie maximale.
- Support des images en entrée.
- Support du raisonnement (thinking) et des paramètres associés s'ils sont déclarés par OpenRouter.
- Support du function calling (toujours considéré comme actif côté requête, voir plus bas).

Ces métadonnées sont figées au moment de l'ajout. Si OpenRouter modifie ses métadonnées plus tard, l'utilisateur peut supprimer puis re-ajouter le modèle pour rafraîchir.

La liste curatée est persistée dans la base SQLite des préférences globales de l'application, au même niveau que les autres réglages globaux (paramètres MCP, paramètres d'outils, etc.). Elle est donc partagée entre toutes les fenêtres et tous les workspaces.

## Gestion de la liste des modèles ajoutés

Sous la zone de recherche, la liste des modèles OpenRouter actifs est affichée en permanence, par ordre d'ajout. Chaque entrée montre le nom du modèle et un bouton "Supprimer" (icône poubelle ou équivalent). Cliquer sur supprimer retire immédiatement le modèle de la liste curatée et donc du sélecteur de modèles partout dans l'application.

Si l'utilisateur supprime un modèle qui était sélectionné dans une conversation, un sub-agent ou un membre d'équipe, la sélection retombe sur le modèle par défaut au prochain démarrage de la conversation ou du turn (comme cela se fait déjà quand un provider est déconnecté).

## Présentation dans le sélecteur de modèles du chat

Les modèles OpenRouter ajoutés apparaissent dans le menu de sélection du modèle (composer du chat) à la suite des modèles natifs des autres providers. Tous, quel que soit le modèle sous-jacent, partagent l'icône et la marque OpenRouter ; l'icône du provider d'origine (Anthropic, Google, etc.) n'est pas affichée.

Le label de chaque entrée est le nom lisible enregistré au moment de l'ajout.

Le sous-menu de niveau de "thinking" associé à un modèle OpenRouter est conditionné par les métadonnées capturées lors de l'ajout :

- Si le modèle déclare supporter le raisonnement, les niveaux off / low / medium / high sont proposés.
- Sinon, le sous-menu thinking est absent ou figé sur "off".

Le support des images se reflète aussi dans les capacités exposées à l'agent (envoi de pièces jointes image autorisé ou non).

## Disponibilité dans le reste de l'application

Les modèles OpenRouter ajoutés sont sélectionnables partout où un modèle peut l'être, exactement comme un modèle natif :

- Dans les trois modes principaux du chat (act, plan, goal).
- Dans la configuration de chaque sub-agent.
- Dans la configuration des membres d'une équipe (swarm / team run).
- Dans toute autre zone existante où un sélecteur de modèles est exposé.

La règle "le provider est configuré" qui filtre actuellement la liste des modèles disponibles est étendue : OpenRouter est considéré comme configuré dès lors qu'une clé API valide est enregistrée, et seuls les modèles présents dans la liste curatée de l'utilisateur sont proposés (et non l'intégralité du catalogue OpenRouter).

## Comportement durant un échange

Quand l'utilisateur envoie un message avec un modèle OpenRouter sélectionné, l'application route la requête vers OpenRouter en utilisant son endpoint compatible OpenAI Chat Completions, avec la clé API placée dans l'en-tête d'authentification standard. Les en-têtes optionnels recommandés par OpenRouter pour identifier l'application appelante (référent et titre du produit) sont ajoutés.

Le streaming des réponses se fait en SSE et est normalisé pour produire les mêmes événements internes que les autres providers (deltas de texte, deltas de raisonnement quand applicable, appels d'outils, fin de message, erreurs, usage de tokens).

Les outils (function calling) sont toujours envoyés dans la requête, comme pour les autres providers, même si certains modèles OpenRouter ne savent pas les exploiter. Les erreurs ou comportements qui en découlent restent à la charge du modèle ciblé.

Le niveau de thinking choisi par l'utilisateur est traduit en paramètres OpenRouter compatibles. Si le modèle ne supporte pas le thinking, le paramètre est simplement omis.

L'estimation du nombre de tokens en entrée (utilisée notamment par les jauges de contexte de l'UI) repose sur une approximation heuristique partagée, à défaut d'un endpoint de comptage natif côté OpenRouter.

## Gestion des erreurs et résilience

- Si une requête échoue parce que la clé est invalide ou révoquée, l'erreur est remontée à l'utilisateur, et l'état de la carte OpenRouter retombe sur "Attention requise" au prochain rafraîchissement de la page Providers.
- Si OpenRouter répond par une erreur de quota, de modèle indisponible ou de routage, le message d'erreur est propagé tel quel dans le chat.
- Si l'utilisateur a sélectionné un modèle OpenRouter et supprime sa clé, les conversations en cours stoppent proprement (via le mécanisme d'annulation existant) et le sélecteur retombe sur un modèle valide au prochain démarrage de turn.

## Hors périmètre de ce plan

Pour rester focalisé, ces sujets ne sont pas couverts ici (à traiter séparément si besoin) :

- Renommer un modèle dans la liste curatée.
- Réordonner la liste par drag & drop.
- Afficher le prix par token, des badges techniques (vision / tools / reasoning) ou l'identifiant OpenRouter dans les résultats de recherche.
- Rafraîchir automatiquement les métadonnées de modèles déjà ajoutés.
- Les routes spécifiques d'OpenRouter au-delà du chat (génération d'images, embeddings, audio, transcriptions).
