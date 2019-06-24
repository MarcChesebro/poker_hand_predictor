"""
Poker hand dataset from uci
https://archive.ics.uci.edu/ml/datasets/Poker+Hand

Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
Attributes:
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
1)  S1 "Suit of card #1" 
2)  C1 "Rank of card #1" 

3)  S2 "Suit of card #2" 
4)  C2 "Rank of card #2" 

5)  S3 "Suit of card #3" 
6)  C3 "Rank of card #3" 

7)  S4 "Suit of card #4" 
8)  C4 "Rank of card #4" 

9)  S5 "Suit of card #5" 
10) C5 "Rank of card #5" 

Class:

11) CLASS "Poker Hand" 
Ordinal (0-9) 

0: Nothing in hand; not a recognized poker hand 
1: One pair; one pair of equal ranks within five cards 
2: Two pairs; two pairs of equal ranks within five cards 
3: Three of a kind; three equal ranks within five cards 
4: Straight; five cards, sequentially ranked with no gaps 
5: Flush; five cards with the same suit 
6: Full house; pair + different rank three of a kind 
7: Four of a kind; four equal ranks within five cards 
8: Straight flush; straight + flush 
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 

accurarcy: 99.28%
"""
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import np_utils, plot_model
from keras.callbacks import TensorBoard

# set seed
numpy.random.seed(10)

class PokerHandPredictor:
	"""
	A simple class for using a trained model to predict poker hands.
	"""

	card_encoding = [
		"Ace",
		"2",
		"3",
		"4",
		"5",
		"6",
		"7",
		"8",
		"9",
		"10",
		"Jack",
		"Queen",
		"King"
	]

	suit_encoding = [
		" of Hearts",
		" of Spades",
		" of Diamonds", 
		" of Clubs"
	]

	output_encoding = [
		"Nothing in hand",
		"One pair",
		"Two pair",
		"Three of a kind",
		"Straight",
		"Flush",
		"Full House",
		"Four of a kind",
		"Straight flush",
		"Royal flush"
	]


	def __init__(self, model_file):
		self.model = load_model(model_file)

	def predict(self, predictions, verbose=False):
		"""
		prediction list in the format:
		Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
		Attributes:
		Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
		1)  S1 "Suit of card #1" 
		2)  C1 "Rank of card #1" 

		3)  S2 "Suit of card #2" 
		4)  C2 "Rank of card #2" 

		5)  S3 "Suit of card #3" 
		6)  C3 "Rank of card #3" 

		7)  S4 "Suit of card #4" 
		8)  C4 "Rank of card #4" 

		9)  S5 "Suit of card #5" 
		10) C5 "Rank of card #5" 
		"""
		output = self.model.predict(numpy.array([predictions]))
		large_index = 0
		large = 0
		for i, out in enumerate(output.tolist()[0]):
			if out > large:
				large = out
				large_index = i

		if verbose:
			print("\nPercentages:")
			for i, out in enumerate(output.tolist()[0]):
				print("{}: {:5.2f}%".format(poker_predictor.output_encoding[i], out * 100))

		return poker_predictor.output_encoding[large_index]

	def verbose_prediction(self, predictions):
		for i, suit in enumerate(predictions):
			if i % 2 == 0:
				print("{}".format(self.decode_input(predictions[i+1], suit)))

		print("\nChoice: " + self.predict(predictions, verbose=True))

	@classmethod
	def decode_input(cls, card, suit):
		return cls.decode_card(card) + cls.decode_suit(suit)

	@classmethod
	def decode_card(cls, card):
		return cls.card_encoding[card - 1]

	@classmethod
	def decode_suit(cls, suit):
		return cls.suit_encoding[suit - 1]


def create_model():
	# Build model
	model = Sequential()	
	model.add(Dense(24, input_dim=10, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

if __name__ == "__main__":


	# This is a very large amount of epochs for this data set but it makes nice tensorboard graphs
	EPOCHS = 1400
	BATCH_SIZE = 12

	# get data
	poker_hands_data = numpy.loadtxt('poker_hand_training.txt', delimiter=",")

	# seperate into inputs and outputs
	poker_hand_input = poker_hands_data[:,0:10]
	raw_output = poker_hands_data[:,10]

	# one hot encoding
	poker_hand_output = np_utils.to_categorical(raw_output)

	# get the base model
	model = create_model()

	# Collect tensorboard data
	tensorboard_checkpoint = TensorBoard(
		log_dir='./logs',
		histogram_freq=0, 
		batch_size=BATCH_SIZE, 
		write_graph=True, 
		write_grads=False, 
		write_images=False, 
		embeddings_freq=0, 
		embeddings_layer_names=None, 
		embeddings_metadata=None, 
		embeddings_data=None, 
		update_freq='epoch'
	)
	callbacks = [tensorboard_checkpoint]

	# Fit the model
	model.fit(poker_hand_input, poker_hand_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
	model.save('poker_model.h5')

	# load and evaluate with testing data.
	poker_testing_data = numpy.loadtxt('poker_hand_testing.txt', delimiter=",")

	# seperate into inputs and outputs
	poker_testing_input = poker_testing_data[:,0:10]
	raw_testing_output = poker_testing_data[:,10]

	# one hot encoding
	poker_testing_output = np_utils.to_categorical(raw_testing_output)

	print("Evaluating model:")

	# evaluate on the testing set
	scores = model.evaluate(poker_testing_input, poker_testing_output)

	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
