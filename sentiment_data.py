"""Hand-written sentiment data and word lists for the Phase 7 and Phase 8 experiments.

Every list is plain text so it can be read and edited without running anything.

- EXPLICIT_*: sentences whose sentiment is stated outright. Used to fit and test directions.
- IMPLICIT_*: sentiment carried only by the situation, with no obvious "feeling" words.
  These are never used to pick a layer, so they are an honest check that the direction
  tracks sentiment and not just a handful of emotion words.
- NEUTRAL_*: sentiment-free text, in four separate groups so no sentence does two jobs:
  CENTER sets the reader's zero point, TURNS are the unrelated interactions in the persistence
  test, DEV picks the steering strength, REPORT measures the final fluency cost.
- TARGET_*: neutral prompts whose next word could plausibly be an opinion ("... was").
"""

from itertools import product

EXPLICIT_POSITIVE = (
    "Something amazing happened to me today.",
    "I absolutely loved the concert last night.",
    "The new cafe on Main Street is wonderful.",
    "She was thrilled to hear the good news.",
    "What a beautiful morning to be alive.",
    "We had a fantastic time at the lake.",
    "The team celebrated a huge victory together.",
    "He felt proud and happy after finishing the race.",
    "This is the best meal I have ever eaten.",
    "My grandmother's visit made the whole house cheerful.",
    "The kids were delighted with their gifts.",
    "I am so grateful for all your help.",
    "The garden looks gorgeous this spring.",
    "It was a joy to work with such a kind group.",
    "Our trip exceeded every expectation we had.",
    "The students cheered when they saw the results.",
    "She smiled warmly and welcomed everyone inside.",
    "I feel wonderful after a long and restful sleep.",
    "The performance was brilliant from start to finish.",
    "He was overjoyed to see his old friends again.",
    "The service at the hotel was excellent.",
    "We are excited about the opportunities ahead.",
    "That movie was hilarious and heartwarming.",
    "The neighbors threw a lovely party for us.",
    "I enjoyed every minute of the hike.",
    "Her kindness made a real difference to everyone.",
    "The bakery smelled delicious and the bread was perfect.",
    "They were pleased with how smoothly everything went.",
    "It was a delightful surprise to receive your letter.",
    "The sunset over the bay was breathtaking.",
    "I am thrilled with my new apartment.",
    "The audience gave a joyful standing ovation.",
    "He laughed happily as the puppy ran circles around him.",
    "The workshop was inspiring and full of great ideas.",
    "Everything about the wedding was magical.",
    "She felt confident and cheerful walking into the interview.",
    "Our team did an outstanding job on the launch.",
    "The weather was perfect and the mood was great.",
    "I love how friendly and warm this town is.",
    "The reunion filled the room with laughter and affection.",
    "It was a fabulous day full of good surprises.",
    "The teacher praised the class for their wonderful effort.",
    "They had a splendid evening dancing under the stars.",
    "I am delighted by how well the project turned out.",
    "Her speech was uplifting and left everyone smiling.",
    "The fresh mountain air made me feel alive and happy.",
    "We enjoyed a relaxing, pleasant afternoon by the river.",
    "The whole family was glad to be together again.",
)

EXPLICIT_NEGATIVE = (
    "Something terrible happened to me today.",
    "I absolutely hated the concert last night.",
    "The new cafe on Main Street is awful.",
    "She was devastated to hear the bad news.",
    "What a miserable morning to be stuck here.",
    "We had a dreadful time at the lake.",
    "The team suffered a crushing defeat together.",
    "He felt ashamed and angry after finishing the race.",
    "This is the worst meal I have ever eaten.",
    "My grandmother's illness made the whole house gloomy.",
    "The kids were disappointed with their gifts.",
    "I am so frustrated by all the delays.",
    "The garden looks dreadful after the frost.",
    "It was a misery to work with such a rude group.",
    "Our trip fell short of every expectation we had.",
    "The students groaned when they saw the results.",
    "She frowned coldly and ignored everyone inside.",
    "I feel awful after a long and sleepless night.",
    "The performance was dismal from start to finish.",
    "He was heartbroken to lose his old friends.",
    "The service at the hotel was appalling.",
    "We are worried about the problems ahead.",
    "That movie was dull and depressing.",
    "The neighbors started a nasty fight with us.",
    "I hated every minute of the hike.",
    "Her cruelty did real damage to everyone.",
    "The bakery smelled rotten and the bread was stale.",
    "They were upset with how badly everything went.",
    "It was a painful shock to receive your letter.",
    "The sky over the bay was bleak and depressing.",
    "I am furious about my new apartment.",
    "The audience gave an angry chorus of boos.",
    "He cried miserably as the puppy ran away from him.",
    "The workshop was tedious and full of bad ideas.",
    "Everything about the wedding was a disaster.",
    "She felt anxious and tearful walking into the interview.",
    "Our team did a terrible job on the launch.",
    "The weather was foul and the mood was grim.",
    "I hate how hostile and cold this town is.",
    "The funeral filled the room with silence and grief.",
    "It was a horrible day full of bad surprises.",
    "The teacher scolded the class for their lazy effort.",
    "They had a wretched evening arguing in the dark.",
    "I am disgusted by how badly the project turned out.",
    "Her speech was bitter and left everyone feeling sick.",
    "The stale smoky air made me feel sick and unhappy.",
    "We endured a tense, unpleasant afternoon by the river.",
    "The whole family was sad to be apart again.",
)

IMPLICIT_POSITIVE = (
    "She got the job she had wanted for years.",
    "The doctor said the test results were completely clear.",
    "After months of practice, he finally played the whole piece without a mistake.",
    "Her daughter ran across the finish line and into her arms.",
    "The bank approved the loan and the keys were handed over.",
    "They found their lost dog waiting on the porch.",
    "The whole class passed, and the teacher shook every hand.",
    "He opened the envelope and saw the acceptance letter.",
    "The crowd rose to its feet and applauded for several minutes.",
    "Her business doubled its sales in the first quarter.",
    "The rain stopped just as the picnic began.",
    "The old friends hugged for a long time at the station.",
    "He was promoted on the same day his son was born.",
    "The surgery went perfectly and she went home that evening.",
    "They danced in the kitchen until the song ended.",
    "The scholarship covered all four years of tuition.",
    "The flowers she planted last autumn came up in bright rows.",
    "He found a twenty-dollar bill in his old coat pocket.",
    "The baby took her first steps toward her father.",
    "Their flight landed early and the bags came out first.",
    "The painting sold for twice the asking price.",
    "She finished the marathon and her family was waiting with a banner.",
    "The landlord lowered the rent and fixed the heater.",
    "His novel was picked up by a major publisher.",
)

IMPLICIT_NEGATIVE = (
    "She was laid off after twenty years at the company.",
    "The doctor said the test results required further treatment.",
    "After months of practice, he forgot the whole piece on stage.",
    "Her daughter walked out and did not call for a year.",
    "The bank denied the loan and the house was sold to someone else.",
    "They never found their lost dog.",
    "Half the class failed, and the teacher shook her head slowly.",
    "He opened the envelope and saw the rejection letter.",
    "The crowd walked out before the second act was over.",
    "Her business lost half its customers in the first quarter.",
    "The rain started just as the picnic began.",
    "The old friends parted at the station without a word.",
    "He was fired on the same day his son was born.",
    "The surgery had complications and she stayed in the hospital for weeks.",
    "They sat in the kitchen in silence until it got dark.",
    "The scholarship was cancelled before the first semester began.",
    "The flowers she planted last autumn were eaten by the frost.",
    "He lost his wallet somewhere between the train and the office.",
    "The baby burned her hand on the stove.",
    "Their flight was cancelled and the bags were lost.",
    "The painting sold for half the asking price, and nobody came to the opening.",
    "She dropped out of the marathon and her family drove home without speaking.",
    "The landlord doubled the rent and ignored the broken heater.",
    "His novel was returned by every publisher he tried.",
)

NEUTRAL_CENTER = (
    "The train leaves from platform four at nine.",
    "Please send the report as a PDF attachment.",
    "The library closes at six on weekdays.",
    "He parked the car next to the red building.",
    "The meeting is scheduled for the third floor.",
    "There are twelve chairs around the table.",
    "She wrote down the address on a card.",
    "The package arrived on Tuesday afternoon.",
    "Water boils at one hundred degrees Celsius.",
    "The bus route passes the post office and the school.",
    "He changed the password and logged out.",
    "The form must be signed in two places.",
    "A new window was installed in the hallway.",
    "The shop sells paper, pens, and envelopes.",
    "They moved the desk closer to the door.",
    "The river runs north for about forty miles.",
    "Her phone number ends in four seven two.",
    "The report has three sections and an appendix.",
    "The ferry crosses the harbor every half hour.",
    "He counted the boxes and wrote the total.",
)

NEUTRAL_TURNS = (
    "The printer is on the second floor next to the supply closet.",
    "Tickets can be collected at the front desk before noon.",
    "The road closes for repairs between the bridge and the park.",
    "She filed the invoices in alphabetical order.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",
    "A new bike lane runs along the east side of the avenue.",
)

NEUTRAL_DEV = (
    "The committee will review the budget proposal on the first of next month.",
    "Rainfall in the valley averages about thirty inches per year.",
    "He transferred the files from the old laptop to an external drive.",
    "The museum is located two blocks north of the central station.",
    "Applicants must submit a copy of their identification with the form.",
    "The software requires at least four gigabytes of memory to run.",
    "A freight train passes the village every morning at half past six.",
    "The survey asked participants about their commute and working hours.",
    "She measured the width of the window before ordering the curtains.",
    "The lecture covers the history of the postal system in three parts.",
    "Orders placed before two in the afternoon are shipped the same day.",
    "The bridge is made of steel and was assembled in several sections.",
    "He set an alarm for seven and left the keys on the table.",
    "The conference room can seat about twenty people.",
    "The tour begins at the main gate and ends at the harbor.",
    "Paper receipts are stored in the cabinet for seven years.",
    "The schedule lists three sessions on Monday and two on Tuesday.",
    "A small team maintains the database and updates it every quarter.",
    "The map shows the main roads and the locations of the public libraries.",
    "The shipment contains forty boxes, each labeled with a number.",
)

NEUTRAL_REPORT = (
    "The council will meet in the town hall on the second Thursday of each month.",
    "Average temperatures in the region rise gradually between March and June.",
    "She copied the contact list into a spreadsheet and sorted it by city.",
    "The pharmacy is open from nine until five on weekdays and closed on Sundays.",
    "Visitors must sign the register at the entrance and collect a badge.",
    "The program stores its settings in a file inside the user directory.",
    "A ferry service connects the island to the mainland several times a day.",
    "The study followed two hundred households over a period of five years.",
    "He wrote the dimensions of the room on the back of an envelope.",
    "The course consists of twelve weekly lectures and a final written exam.",
    "Deliveries to the warehouse are accepted on weekday mornings only.",
    "The tower was built in the nineteenth century using local limestone.",
    "She placed the umbrella by the door and hung her coat on the hook.",
    "The auditorium has four hundred seats arranged in fifteen rows.",
    "The trail follows the river for six miles before turning toward the hills.",
    "Meeting minutes are posted on the shared drive within two days.",
    "The timetable shows departures every twenty minutes during the day.",
    "A volunteer group catalogs the archive and scans a few pages each week.",
    "The atlas includes maps of each province and a list of major cities.",
    "The invoice lists the item numbers, quantities, and unit prices.",
)

# Neutral frames x subjects. Subjects are split so test prompts use subjects never seen
# while choosing layers or strengths.
TARGET_FRAMES = (
    "{subject} today was",
    "Overall, {subject} turned out to be",
    "Honestly, I thought {subject} was",
    "Looking back, {subject} felt",
)
DEV_SUBJECTS = ("the weather", "the meeting", "the trip")
TEST_SUBJECTS = (
    "my new apartment", "the restaurant", "the movie", "her speech", "the hiking trail",
    "the software update", "the conference", "the neighborhood", "the exam",
)

# Words whose next-token probability defines "how positive is the continuation".
POSITIVE_WORDS = (
    "good", "great", "wonderful", "happy", "lovely", "nice", "amazing", "pleasant",
    "fantastic", "excellent", "beautiful", "delightful", "perfect", "fun",
)
NEGATIVE_WORDS = (
    "bad", "terrible", "awful", "sad", "horrible", "dreadful", "miserable", "unpleasant",
    "disappointing", "poor", "ugly", "dull", "boring", "painful",
)

# Wider stems used to judge generated text (matched on word beginnings).
LEXICON_POSITIVE = (
    "good", "great", "wonderful", "happy", "love", "nice", "amazing", "pleasant", "fantastic",
    "excellent", "beautiful", "delight", "perfect", "fun", "enjoy", "joy", "glad", "best",
    "smile", "laugh", "success", "win", "thank", "proud", "bright", "warm",
)
LEXICON_NEGATIVE = (
    "bad", "terrible", "awful", "sad", "horrible", "dread", "miserable", "unpleasant",
    "disappoint", "poor", "ugly", "dull", "boring", "pain", "hate", "worst", "angry", "fear",
    "cry", "fail", "lose", "lost", "sick", "dark", "cold", "grim", "wrong", "die", "death",
)


def targets(subjects: tuple[str, ...]) -> list[str]:
    return [frame.format(subject=subject) for subject, frame in product(subjects, TARGET_FRAMES)]


DEV_TARGETS = tuple(targets(DEV_SUBJECTS))
TEST_TARGETS = tuple(targets(TEST_SUBJECTS))
