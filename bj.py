module = "bj"
import random


class BlackjackGame:
    def __init__(self, state):
        self.deck = self.create_deck()
        self.player_hand = self.create_hand_player(state[0], 12, state[2])
        self.dealer_hand = self.create_hand_dealer(state[1], 1)
        self.game_over = False

    def create_deck(self):
        """Create and shuffle a deck of 52 cards."""
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        deck = [{'suit': suit, 'value': value} for suit in suits for value in values]
        random.shuffle(deck)
        return deck

    def card_value(self, card):
        """Return the value of a single card."""
        if card['value'] in ['Jack', 'Queen', 'King']:
            return 10
        elif card['value'] == 'Ace':
            return 11  # Value of Ace can be 1 or 11, this will be handled in total_hand
        else:
            return int(card['value'])

    def total_hand(self, hand, max=21):
        """Return the total value of a hand of cards, adjust for Aces as necessary."""
        total = sum(self.card_value(card) for card in hand)
        # Adjust for Aces
        num_aces = sum(1 for card in hand if card['value'] == 'Ace')
        while total > max and num_aces:
            total -= 10
            num_aces -= 1
        return total

    def hit(self, hand_name):
        """Add a card to a hand."""
        if hand_name == "player":
            self.player_hand.append(self.deck.pop())
            if self.total_hand(self.player_hand) > 21:
                self.game_over = True
        elif hand_name == "dealer":
            self.dealer_hand.append(self.deck.pop())

    def stand(self):
        """Handle the dealer's moves and conclude the game."""
        while self.total_hand(self.dealer_hand) < 17:
            self.hit("dealer")
        self.game_over = True

    def start_game(self):
        """Start a new game using the initial hands if provided, otherwise deal new cards."""
        if not self.player_hand or not self.dealer_hand:
            self.player_hand = [self.deck.pop(), self.deck.pop()]
            self.dealer_hand = [self.deck.pop(), self.deck.pop()]

    def has_usable_ace(self, hand):
        """Check if the hand has a usable Ace."""
        return 'Ace' in [card['value'] for card in hand] and self.total_hand(hand) <= 21

    def get_game_state(self):
        """Return the current state of the game suitable for Monte Carlo ES."""
        player_total = self.total_hand(self.player_hand)
        player_usable_ace = self.has_usable_ace(self.player_hand)
        # Assuming the first card of the dealer is the showing card
        dealer_showing_card_value = self.card_value(self.dealer_hand[0])
        # Convert face cards to numeric values for the state representation
        if dealer_showing_card_value > 10:
            dealer_showing_card_value = 10  # Face cards are worth 10
        state_ = {
            "player_total": player_total,
            "player_usable_ace": player_usable_ace,
            "dealer_showing_card": dealer_showing_card_value,
        }
        return state_

    def create_hand_player(self, value, value_offset, usable_ace=False):
        value += value_offset
        # Generate one random card
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        if not usable_ace:
            while True:
                card = {'suit': random.choice(suits), 'value': random.choice(values)}
                card2 = {'suit': random.choice(suits), 'value': random.choice(values)}
                hand = [card, card2]
                hand_value = self.total_hand(hand)
                if hand_value == value:
                    print(hand, hand_value)
                    return [card, card2]
        else:
            while True:
                card = {'suit': random.choice(suits), 'value': 'Ace'}
                card2 = {'suit': random.choice(suits), 'value': random.choice(values)}
                hand = [card, card2]
                hand_value = self.total_hand(hand)
                if hand_value == value:
                    print(hand, hand_value)
                    return [card, card2]

    def create_hand_dealer(self, value, value_offset):
        value += value_offset
        # Generate one random card
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        card2 = {'suit': random.choice(suits), 'value': random.choice(values)}
        while True:
            card = {'suit': random.choice(suits), 'value': random.choice(values)}
            if (self.card_value(card) == value) or (value == 1 and self.card_value(card) == 11):
                print(card, card2)
                return [card, card2]

    def get_winner(self):
        player_total = self.total_hand(self.player_hand)
        dealer_total = self.total_hand(self.dealer_hand)
        if player_total > 21:
            return -1
        elif dealer_total > 21:
            return 1
        elif player_total > dealer_total:
            return 1
        elif player_total < dealer_total:
            return -1
        else:
            return 0
