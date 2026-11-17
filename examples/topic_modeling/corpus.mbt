// Fixed, reproducible corpus for the topic modeling MVP.
//
// `expected_labels` are human references used only for interpretation in the
// README. They are never fed into tokenization, the vocabulary, V, or NMF
// training. Topic order in the output is NOT aligned with these labels.

///|
let training_docs : Array[String] = [
  // Technology (docs 0-4; doc 4 mixes in sports words)
  "computer software data network processor digital", "software data digital computer network software",
  "processor computer digital data network processor", "network software computer data digital network",
  "computer data network player team match digital",
  // Sports (docs 5-9; doc 9 mixes in cooking words)
   "player team match coach score stadium", "team player coach match score team",
  "stadium match player score coach stadium", "coach team score match player coach",
  "player match stadium recipe kitchen food score",
  // Cooking (docs 10-14; doc 14 mixes in technology words)
   "recipe kitchen food flavor oven ingredient", "kitchen recipe ingredient food flavor kitchen",
  "oven food flavor recipe ingredient oven", "ingredient kitchen flavor food recipe ingredient",
  "recipe food oven software data kitchen flavor",
]

///|
let expected_labels : Array[String] = [
  "technology", "technology", "technology", "technology", "technology+sports", "sports",
  "sports", "sports", "sports", "sports+cooking", "cooking", "cooking", "cooking",
  "cooking", "cooking+technology",
]

///|
let new_docs : Array[String] = [
  "computer network digital router cloud software", "team coach stadium referee match whistle player",
  "kitchen oven ingredient microwave recipe simmer food",
]
