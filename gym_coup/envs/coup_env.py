import gym
import numpy as np
from random import shuffle
import logging

logging.basicConfig()
logger = logging.getLogger('gym_coup')

NONE = -1

# Cards
ASSASSIN   = 0
AMBASSADOR = 1
CAPTAIN    = 2
CONTESSA   = 3
DUKE       = 4

# Actions
INCOME                      = 0
FOREIGN_AID                 = 1
COUP                        = 2
TAX                         = 3
ASSASSINATE                 = 4
EXCHANGE                    = 5
STEAL                       = 6
LOSE_CARD_1                 = 7
LOSE_CARD_2                 = 8
PASS_FA                     = 9
PASS_FA_BLOCK               = 10
PASS_TAX                    = 11
PASS_EXCHANGE               = 12
PASS_ASSASSINATE_BLOCK      = 13
PASS_STEAL                  = 14
PASS_STEAL_BLOCK            = 15
BLOCK_FA                    = 16
BLOCK_ASSASSINATE           = 17
BLOCK_STEAL                 = 18
CHALLENGE_FA_BLOCK          = 19
CHALLENGE_TAX               = 20
CHALLENGE_EXCHANGE          = 21
CHALLENGE_ASSASSINATE       = 22
CHALLENGE_ASSASSINATE_BLOCK = 23
CHALLENGE_STEAL             = 24
CHALLENGE_STEAL_BLOCK       = 25
EXCHANGE_RETURN_12          = 26
EXCHANGE_RETURN_13          = 27
EXCHANGE_RETURN_14          = 28
EXCHANGE_RETURN_23          = 29
EXCHANGE_RETURN_24          = 30
EXCHANGE_RETURN_34          = 31

class Card:
    names = ['Assassin',
             'Ambassador',
             'Captain',
             'Contessa',
             'Duke']
    def __init__(self, val, is_face_up=False):
        self.val = val
        self.is_face_up = is_face_up

    def get_name(self):
        return Card.names[self.val]

    def __lt__(self, other):
        return (self.val < other.val or
                (self.val == other.val and self.is_face_up < other.is_face_up))

class Player:
    def __init__(self, id, is_human=False):
        self.id = id
        self.is_human = is_human
        self.cards = []
        self.coins = 2
        self.last_action = NONE

        # Indicate that the player has lost a challenge
        # and must choose which card to lose
        self.lost_challenge = False

    def add_card(self, card):
        self.cards.append(card)

    def add_coins(self, num):
        self.coins += num

    def remove_coins(self, num):
        self.coins -= num

    def has_face_down_card(self, card_val):
        for i in range(len(self.cards)):
            c = self.cards[i]
            if c.val == card_val and not c.is_face_up:
                return True
        return False

    def get_obs(self, text=False):
        '''
        Return the current state of the player

        text: Whether to get in readable format

        Observation:
            Cards list [(
                Card            (0 - 4)
                Is card face up (0 - 1)
                )]
            Coins               (0 - 12)
            Last action         (-1 - 31)
        '''
        if text:
            c = [(self.cards[i].get_name(), int(self.cards[i].is_face_up)) for i in range(len(self.cards))]
            la = CoupEnv.actions[self.last_action]
        else:
            c = [(self.cards[i].val, int(self.cards[i].is_face_up)) for i in range(len(self.cards))]
            la = self.last_action
        return (c,
                self.coins,
                la)

    def _sort_cards(self):
        '''
        Always keep the cards sorted in alphabetical order.
        Since get_obs is run on each iter, decrease the amount of
        sorts we need by only sorting when cards are exchanged or lost.
        '''
        self.cards.sort()

    def render(self):
        text = f'P{self.id + 1}: '
        for c in self.cards:
            text += f'{c.get_name()} '
        text += '| '
        for c in self.cards:
            text += f'{c.is_face_up} '
        text += '| '
        text += f'{self.coins} | {"_" if self.last_action == NONE else CoupEnv.actions[self.last_action]}'
        logger.info(text)


class Game:
    '''
    2 player Coup game
    Can have any combination of human and cpu players
    '''
    def __init__(self, num_human_players=0, p_first_turn=0):
        '''
        num_human_players: Number of human players in the 2-player game
        p_first_turn:      Which player goes first, 0-indexed
        '''
        self.players = [Player(i, True) for i in range(num_human_players)]
        self.players += [Player(i+num_human_players, False) for i in range(2-num_human_players)]

        self.deck = [Card(i) for _ in range(3) for i in range(len(Card.names))]
        self.shuffle_deck()
        self.deal_cards()

        # P1 = 0, P2 = 1
        # Whose overall turn is it?
        self.whose_turn = p_first_turn
        # Turns can include several sub-actions.
        # Who is currently choosing an action?
        self.whose_action = p_first_turn

        self.turn_count = 0

        # Is it the beginning of a new turn?
        self.is_turn_begin = True

        self.game_over = False

        if len(self.players) == 2:
            # In a 2 player game, the player going first starts with 1 coin instead of 2
            self.players[p_first_turn].coins = 1

    def get_obs(self, p2_view=False, text=False):
        '''
        Return the current state of the game

        p2_view: Whether to get the observation from P2's view/perspective
        text:    Whether to get in readable format

        Observation:
            P1 Cards list [(
                Card            (0 - 4)
                Is card face up (0 - 1)
                )]
            P2 Cards list [(
                Card            (0 - 4)
                Is card face up (0 - 1)
                )]
            P1 # coins           (0 - 12)
            P2 # coins           (0 - 12)
            P1 last action       (-1 - 31)
            P2 last action       (-1 - 31)
            Whose next action    (0 - 1)
        '''
        if p2_view:
            p1_ind = 1
            p2_ind = 0
        else:
            p1_ind = 0
            p2_ind = 1
        p1 = self.players[p1_ind].get_obs(text=text)
        p2 = self.players[p2_ind].get_obs(text=text)
        return (p1[0], p2[0], # Cards
                p1[1], p2[1], # Coins
                p1[2], p2[2], # Last action
                self.whose_action)

    def render(self):
        logger.info(f'Turn {self.turn_count}')
        logger.info('Player: Cards | IsCardFaceUp | Coins | LastAction')
        for p in self.players:
            p.render()

    def draw_card(self, index=0):
        return self.deck.pop(index)

    def shuffle_deck(self):
        shuffle(self.deck)

    def deal_cards(self):
        for _ in range(2):
            for p in self.players:
                p.add_card(self.draw_card())
        self.players[0]._sort_cards()
        self.players[1]._sort_cards()

    def next_player_turn(self):
        '''
        Increment whose turn it is
        Turns can include several sub-actions
            ex: P1 Steal, P2 Block, P1 Challenge
                is a single turn of 3 actions
        '''
        self.whose_turn = 1 - self.whose_turn
        # Players will always have the first action on their turn
        self.whose_action = self.whose_turn
        self.turn_count += 1
        self.is_turn_begin = True

    def next_player_action(self):
        '''
        Increment whose action it is
        '''
        self.whose_action = 1 - self.whose_action
        self.is_turn_begin = False

    def get_curr_action_player(self):
        return self.players[self.whose_action]

    def get_opp_player(self):
        return self.players[1 - self.whose_action]

    def get_valid_actions(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()

        def valid_lose_card_options():
            valid = []
            if not curr_player.cards[0].is_face_up:
                # Card is still in play. Can choose to give it up.
                valid += [LOSE_CARD_1]
            if not curr_player.cards[1].is_face_up:
                # Card is still in play. Can choose to give it up.
                valid += [LOSE_CARD_2]
            return valid


        if self.is_turn_begin:
            # It's the beginning of curr_player's turn

            if curr_player.coins >= 10:
                return [COUP]

            valid = [INCOME, FOREIGN_AID, TAX, EXCHANGE]
            if curr_player.coins >= 3:
                valid.append(ASSASSINATE)
            if curr_player.coins >= 7:
                valid.append(COUP)
            if opp_player.coins > 0:
                valid.append(STEAL)
            
            return valid

        elif curr_player.lost_challenge:
            return valid_lose_card_options()

        elif self.whose_turn != self.whose_action:
            # It is opp_player's turn, and curr_player can
            # choose to block or challenge for certain actions

            if opp_player.last_action == FOREIGN_AID:
                return [PASS_FA, BLOCK_FA]
            elif opp_player.last_action == TAX:
                return [PASS_TAX, CHALLENGE_TAX]
            elif opp_player.last_action == EXCHANGE:
                return [PASS_EXCHANGE, CHALLENGE_EXCHANGE]
            elif opp_player.last_action == STEAL:
                return [PASS_STEAL, BLOCK_STEAL, CHALLENGE_STEAL]
            elif opp_player.last_action in [ASSASSINATE, COUP]:
                valid = valid_lose_card_options()

                if opp_player.last_action == ASSASSINATE:
                    valid += [BLOCK_ASSASSINATE, CHALLENGE_ASSASSINATE]

                return valid
            else:
                raise RuntimeError('Invalid action progression')

        elif curr_player.last_action == EXCHANGE:
            # It is curr_player's turn, and opp_player has approved the exchange

            if len(curr_player.cards) < 4:
                raise RuntimeError('Player mid-exchange should have 4 cards including any eliminated')
            
            valid = [EXCHANGE_RETURN_34]
            if not curr_player.cards[0].is_face_up:
                valid += [EXCHANGE_RETURN_13, EXCHANGE_RETURN_14]
            if not curr_player.cards[1].is_face_up:
                valid += [EXCHANGE_RETURN_23, EXCHANGE_RETURN_24]
            if (not curr_player.cards[0].is_face_up and
                not curr_player.cards[1].is_face_up):
                valid += [EXCHANGE_RETURN_12]

            return valid

        elif opp_player.last_action == BLOCK_FA:
            # It is curr_player's turn and opp_player wants to block their move
            return [PASS_FA_BLOCK, CHALLENGE_FA_BLOCK]
        elif opp_player.last_action == BLOCK_ASSASSINATE:
            # It is curr_player's turn and opp_player wants to block their move
            return [PASS_ASSASSINATE_BLOCK, CHALLENGE_ASSASSINATE_BLOCK]
        elif opp_player.last_action == BLOCK_STEAL:
            # It is curr_player's turn and opp_player wants to block their move
            return [PASS_STEAL_BLOCK, CHALLENGE_STEAL_BLOCK]
        
        else:
            raise RuntimeError('Invalid action progression')


    def income(self):
        curr_player = self.get_curr_action_player()
        curr_player.add_coins(1)
        curr_player.last_action = INCOME
        self.next_player_turn()

    def foreign_aid(self):
        if self.is_turn_begin:
            # Before allowing the action to take effect, the opponent must not block it
            self.get_curr_action_player().last_action = FOREIGN_AID
            self.next_player_action()
        else:
            # PASS: Opponent did not block, so complete the action
            self.get_curr_action_player().add_coins(2)
            self.next_player_turn()

    def coup(self):
        curr_player = self.get_curr_action_player()
        if curr_player.coins < 7:
            raise RuntimeError('Not possible to coup with < 7 coins')

        curr_player.remove_coins(7)
        curr_player.last_action = COUP
        self.next_player_action()

    def tax(self):
        if self.is_turn_begin:
            # Before allowing the action to take effect, the opponent must not challenge it
            self.get_curr_action_player().last_action = TAX
            self.next_player_action()
        else:
            # PASS: Opponent did not challenge, so complete the action
            self.get_curr_action_player().add_coins(3)
            self.next_player_turn()

    def assassinate(self):
        curr_player = self.get_curr_action_player()
        curr_player.last_action = ASSASSINATE
        # Pay the coins whether or not the action is blocked/challenged
        curr_player.remove_coins(3)
        self.next_player_action()

    def exchange(self):
        if self.is_turn_begin:
            # Before drawing the 2 cards from the deck, the opponent must not challenge it
            self.get_curr_action_player().last_action = EXCHANGE
            self.next_player_action()
        else:
            # PASS: Opponent did not challenge, so draw 2 cards
            # CHALLENGE: curr_player had the ambassador, so complete the action
            curr_player = self.get_curr_action_player()
            curr_player.add_card(self.draw_card())
            curr_player.add_card(self.draw_card())
            # Don't increment turn or action
            # It is still curr_player's choice of which cards to return to the deck

    def _exchange_return(self, lst):
        curr_player = self.get_curr_action_player()
        for ind in sorted(lst, reverse=True):
            self.deck.append(curr_player.cards.pop(ind))
        self.shuffle_deck()
        curr_player._sort_cards()

        if self.get_opp_player().lost_challenge:
            # opp still needs to choose a card to lose
            self.next_player_action()
        else:
            self.next_player_turn()

    def exchange_return_12(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_12
        self._exchange_return([0, 1])

    def exchange_return_13(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_13
        self._exchange_return([0, 2])

    def exchange_return_14(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_14
        self._exchange_return([0, 3])

    def exchange_return_23(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_23
        self._exchange_return([1, 2])

    def exchange_return_24(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_24
        self._exchange_return([1, 3])

    def exchange_return_34(self):
        self.get_curr_action_player().last_action = EXCHANGE_RETURN_34
        self._exchange_return([2, 3])

    def steal(self):
        if self.is_turn_begin:
            # Before allowing the action to take effect, the opponent must not block or challenge
            self.get_curr_action_player().last_action = STEAL
            self.next_player_action()
        else:
            # PASS: Opponent did not challenge, so complete the action
            curr_player = self.get_curr_action_player()
            opp_player = self.get_opp_player()

            num_steal = 2 if opp_player.coins >= 2 else 1
            opp_player.remove_coins(num_steal)
            curr_player.add_coins(num_steal)
            self.next_player_turn()

    def _pass(self):
        # Complete the opponent's action
        act = self.get_opp_player().last_action
        self.next_player_action()
        getattr(self, CoupEnv.actions[act])()

    def _pass_block(self):
        # Block succeeds, so nothing to do. Next turn.
        self.next_player_turn()

    def pass_fa(self):
        self.get_curr_action_player().last_action = PASS_FA
        self._pass()

    def pass_fa_block(self):
        self.get_curr_action_player().last_action = PASS_FA_BLOCK
        self._pass_block()

    def pass_tax(self):
        self.get_curr_action_player().last_action = PASS_TAX
        self._pass()

    def pass_exchange(self):
        self.get_curr_action_player().last_action = PASS_EXCHANGE
        self._pass()

    def pass_assassinate_block(self):
        self.get_curr_action_player().last_action = PASS_ASSASSINATE_BLOCK
        self._pass_block()

    def pass_steal(self):
        self.get_curr_action_player().last_action = PASS_STEAL
        self._pass()

    def pass_steal_block(self):
        self.get_curr_action_player().last_action = PASS_STEAL_BLOCK
        self._pass_block()

    def block_fa(self):
        self.get_curr_action_player().last_action = BLOCK_FA
        self.next_player_action()

    def block_assassinate(self):
        self.get_curr_action_player().last_action = BLOCK_ASSASSINATE
        self.next_player_action()

    def block_steal(self):
        self.get_curr_action_player().last_action = BLOCK_STEAL
        self.next_player_action()

    # Challenge:
    # Check if opp_player has the required card
    # If they do, curr_player loses a card
    # If they don't, opp_player loses a card

    def challenge_fa_block(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_FA_BLOCK

        if opp_player.has_face_down_card(DUKE):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(DUKE)
            # curr_player must lose a card
            # It is still their action
        else:
            opp_player.lost_challenge = True

            # Block failed, so complete the action
            curr_player.add_coins(2)

            # opp_player must lose a card
            self.next_player_action()

    def challenge_tax(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_TAX

        if opp_player.has_face_down_card(DUKE):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(DUKE)

            # Complete the action
            opp_player.add_coins(3)

            # curr_player must lose a card
            # It is still their action
        else:
            opp_player.lost_challenge = True
            # opp_player must lose a card
            self.next_player_action()

    def challenge_exchange(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_EXCHANGE

        if opp_player.has_face_down_card(AMBASSADOR):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(AMBASSADOR)

            # Complete the action
            self.next_player_action()
            self.exchange()

            # curr_player must lose a card
            # After _exchange_return is called it will switch to their action
        else:
            opp_player.lost_challenge = True
            # opp_player must lose a card
            self.next_player_action()

    def challenge_assassinate(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_ASSASSINATE

        if opp_player.has_face_down_card(ASSASSIN):
            # curr_player loses the game
            # Lose 1 card for assassination
            # and 1 card for losing challenge
            curr_player.cards[0].is_face_up = True
            curr_player.cards[1].is_face_up = True
            self.game_over = True
            logger.info('Game Over')
        else:
            opp_player.lost_challenge = True

            # Coins spent are returned in this one case
            opp_player.add_coins(3)

            # opp_player must lose a card
            self.next_player_action()

    def challenge_assassinate_block(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_ASSASSINATE_BLOCK

        if opp_player.has_face_down_card(CONTESSA):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(CONTESSA)
            # curr_player must lose a card
            # It is still their action
        else:
            # opp_player loses the game
            # Lose 1 card for assassination
            # and 1 card for losing challenge
            opp_player.cards[0].is_face_up = True
            opp_player.cards[1].is_face_up = True
            self.game_over = True
            logger.info('Game Over')

    def challenge_steal(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_STEAL

        if opp_player.has_face_down_card(CAPTAIN):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(CAPTAIN)

            # Complete the action
            num_steal = 2 if curr_player.coins >= 2 else 1
            curr_player.remove_coins(num_steal)
            opp_player.add_coins(num_steal)
            # curr_player must lose a card
            # It is still their action
        else:
            opp_player.lost_challenge = True
            # opp_player must lose a card
            self.next_player_action()

    def challenge_steal_block(self):
        curr_player = self.get_curr_action_player()
        opp_player = self.get_opp_player()
        curr_player.last_action = CHALLENGE_STEAL_BLOCK

        if opp_player.has_face_down_card(CAPTAIN):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(CAPTAIN)
            # curr_player must lose a card
            # It is still their action
        elif opp_player.has_face_down_card(AMBASSADOR):
            curr_player.lost_challenge = True
            # Replace the revealed card
            self._challenge_fail_replace_card(AMBASSADOR)
            # curr_player must lose a card
            # It is still their action
        else:
            opp_player.lost_challenge = True

            # Block failed, so complete the action
            num_steal = 2 if opp_player.coins >= 2 else 1
            opp_player.remove_coins(num_steal)
            curr_player.add_coins(num_steal)

            # opp_player must lose a card
            self.next_player_action()

    def _challenge_fail_replace_card(self, card_val):
        # If the challenged player actually had the correct card,
        # shuffle it into the deck and give them a new card
        logger.info(f'Showing and replacing card {Card.names[card_val]}')
        p = self.get_opp_player()
        for i in range(len(p.cards)):
            c = p.cards[i]
            if c.val == card_val and not c.is_face_up:
                self.deck.append(c)
                self.shuffle_deck()
                p.cards[i] = self.draw_card()
                p._sort_cards()
                return

        raise RuntimeError(f'Tried to replace card {Card.names[card_val]} that was not in player\'s hand')

    def _lose_card(self, card_ind):
        curr_player = self.get_curr_action_player()
        if curr_player.cards[card_ind].is_face_up:
            raise RuntimeError(f'Cannot lose a card that is already face up')

        curr_player.cards[card_ind].is_face_up = True
        curr_player.lost_challenge = False
        curr_player._sort_cards()

        # Check if the player has no cards remaining
        self.game_over = not (False in [x.is_face_up for x in curr_player.cards])

        if self.game_over:
            logger.info('Game Over')

        self.next_player_turn()

    def lose_card_1(self):
        self.get_curr_action_player().last_action = LOSE_CARD_1
        self._lose_card(0)

    def lose_card_2(self):
        self.get_curr_action_player().last_action = LOSE_CARD_2
        self._lose_card(1)



class CoupEnv(gym.Env):
    '''
    Gym env wrapper for a 2p Coup game
    '''
    metadata = {'render.modes': ['human']}

    actions = {
        -1: 'none',
        0:  'income',
        1:  'foreign_aid',
        2:  'coup',
        3:  'tax',
        4:  'assassinate',
        5:  'exchange', # pick up 2 cards from court deck
        6:  'steal',
        7:  'lose_card_1', # choose which card to lose
        8:  'lose_card_2',
        9:  'pass_fa',
        10: 'pass_fa_block',
        11: 'pass_tax',
        12: 'pass_exchange',
        13: 'pass_assassinate_block',
        14: 'pass_steal',
        15: 'pass_steal_block',
        16: 'block_fa',
        17: 'block_assassinate',
        18: 'block_steal',
        19: 'challenge_fa_block',
        20: 'challenge_tax',
        21: 'challenge_exchange',
        22: 'challenge_assassinate',
        23: 'challenge_assassinate_block',
        24: 'challenge_steal',
        25: 'challenge_steal_block',
        26: 'exchange_return_12', # return cards 1,2 to court deck
        27: 'exchange_return_13', # return cards 1,3
        28: 'exchange_return_14', # return cards 1,4
        29: 'exchange_return_23', # return cards 2,3
        30: 'exchange_return_24', # return cards 2,4
        31: 'exchange_return_34'  # return cards 3,4
    }

    def __init__(self, num_human_players=0, p_first_turn=0):
        '''
        num_human_players: Number of human players in the 2-player game
        p_first_turn:      Which player goes first, 0-indexed
        '''
        self.num_human_players = num_human_players
        self.p_first_turn = p_first_turn
        self.game = None

        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Observation:
        #     P1 card 1            (0 - 4)
        #     P1 card 2            (0 - 4)
        #     P1 card 3            (-1 - 4) # Can have 2 or 4 cards
        #     P1 card 4            (-1 - 4)
        #     P2 card 1            (0 - 4)
        #     P2 card 2            (0 - 4)
        #     P2 card 3            (-1 - 4) # Can have 2 or 4 cards
        #     P2 card 4            (-1 - 4)
        #     P1 is card 1 face up (0 - 1)
        #     P1 is card 2 face up (0 - 1)
        #     P1 is card 3 face up (-1 - 1)
        #     P1 is card 4 face up (-1 - 1)
        #     P2 is card 1 face up (0 - 1)
        #     P2 is card 2 face up (0 - 1)
        #     P2 is card 3 face up (-1 - 1)
        #     P2 is card 4 face up (-1 - 1)
        #     P1 # coins           (0 - 12)
        #     P2 # coins           (0 - 12)
        #     P1 last action       (-1 - 31)
        #     P2 last action       (-1 - 31)
        #     Whose next action    (0 - 1)
        # Note: many observations will never occur in game
        #       ex: All 4 cards are the same. Both players have all cards face up.
        low  = np.array([0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0], dtype='int8')
        high = np.array([4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 31, 31, 1], dtype='int8')
        self.observation_space = gym.spaces.Box(low, high, dtype='int8')

    def step(self, action):
        if isinstance(action, int):
            action = self.actions[action]
        elif isinstance(action, str):
            pass
        else:
            raise RuntimeError(f'Cannot step with action type {type(action)}')

        # Who takes this action
        whose_a = self.game.whose_action

        # Num face up cards of each player before the action
        num_cards_1 = [len([1 for c in p.cards if c.is_face_up]) for p in self.game.players]

        getattr(self.game, action)()

        # Get the observation from the perspective of
        # the player who just took the action
        obs = self.get_obs(whose_a == 1)
        logger.debug(f'Observation: {obs}')

        # Num face up cards of each player after the action
        num_cards_2 = [len([1 for c in p.cards if c.is_face_up]) for p in self.game.players]

        if 2 in num_cards_2 and not self.game.game_over:
            raise RuntimeError('Game over was not set when it should be')

        reward = 0
        # Get number of cards lost this action by each player
        dif_curr = num_cards_2[whose_a] - num_cards_1[whose_a]
        dif_opp = num_cards_2[1-whose_a] - num_cards_1[1-whose_a]
        # -1 if you lose a card
        reward += -1 * dif_curr
        # +1 if your opp loses a card
        reward += 1 * dif_opp
        logger.debug(f'Reward: {reward}')

        return (obs, reward, self.game.game_over, dict())

    def reset(self):
        self.game = Game(self.num_human_players, self.p_first_turn)

    def render(self, mode='human'):
        if self.game is not None:
            self.game.render()

    def get_valid_actions(self, text=False):
        '''
        Get the valid actions, in either number or text form
        '''
        if self.game is None:
            return None

        a = self.game.get_valid_actions()
        logger.debug(f'Valid actions: {[self.actions[x] for x in a]}')
        if text:
            return [self.actions[x] for x in a]
        else:
            return a

    def get_obs(self, p2_view=False, text=False):
        '''
        Return the current state of the environment

        p2_view: Whether to get the observation from P2's view/perspective
        text:    Whether to get in readable format

        Observation:
            P1 card 1            (0 - 4)
            P1 card 2            (0 - 4)
            P1 card 3            (-1 - 4) # Can have 2 or 4 cards
            P1 card 4            (-1 - 4)
            P2 card 1            (0 - 4)
            P2 card 2            (0 - 4)
            P2 card 3            (-1 - 4) # Can have 2 or 4 cards
            P2 card 4            (-1 - 4)
            P1 is card 1 face up (0 - 1)
            P1 is card 2 face up (0 - 1)
            P1 is card 3 face up (-1 - 1)
            P1 is card 4 face up (-1 - 1)
            P2 is card 1 face up (0 - 1)
            P2 is card 2 face up (0 - 1)
            P2 is card 3 face up (-1 - 1)
            P2 is card 4 face up (-1 - 1)
            P1 # coins           (0 - 12)
            P2 # coins           (0 - 12)
            P1 last action       (-1 - 31)
            P2 last action       (-1 - 31)
            Whose next action    (0 - 1)
        Note: many observations will never occur in game
              ex: All 4 cards are the same. Both players have all cards face up.
        '''
        p1cards, p2cards, p1coins, p2coins, p1la, p2la, wa = self.game.get_obs(p2_view=p2_view, text=text)

        obs = []

        # Add cards and is_face_up vals
        for i in range(2):
            for x in [p1cards, p2cards]:
                obs += [c[i] for c in x]
                if len(x) == 2:
                    if text:
                        obs += [self.actions[NONE]]*2
                    else:
                        obs += [NONE]*2
                elif len(x) != 4:
                    raise RuntimeError('Number of cards in hand must be 2 or 4')

        obs += [p1coins, p2coins, p1la, p2la, wa]
        obs = tuple(obs)
        return obs
