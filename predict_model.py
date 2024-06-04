import torch

from utils.predict import predict
from utils.model import NLP

text_1 = '''A Festa de Léo é bom, independentemente dos fatores.
É um filme sensível, feito com o olhar de dentro, das milhares de pessoas que moram nas
periferias dos centros urbanos e que todo dia é dia de luta,
quase nunca um dia de glória – pois, quando se tenta ter um dia de glória,
algo (um Dudu) ocorre para mudar o cenário.
Quando olhamos o argumento do longa, podemos pensar que o mote facilmente poderia ser um curta
– a história de uma mulher que queria dar uma festa pro filho, mas há um desvio na sua
rotina por causa das escolhas do marido –, mas, como o próprio longa comprova,
quando esse mote de um curta acontece no cotidiano periférico, inúmeros elementos concorrem
e ocorrem na vida dos personagens, dando pano para cada dia do cotidiano se transformar num filme,
tamanho o desafio de se chegar ao fim e colocar a cabeça tranquila no travesseiro.'''
#Crítica "A Festa de Léo" - Janda Montenegro - cinepop.com.br

text_2 = '''Por fim, a direção amadora de Mike Burns (que anteriormente trabalhava com a música dos filmes)
é de tranquilizar qualquer estudante de cinema. O trabalho dele é tão ruim, que para fazer a transição das cenas o único
recurso que ele conhece é o fade out (aquele escurecer da tela), e daí temos uma hora e meia de um monte de fade out.
Há uma cena literalmente com a câmera caída na cintura de Chloe, que parece um erro de gravação, mas que entrou no corte final.
Há cenas de câmera tremendo. Há duas cenas de Bruce Willis correndo pela floresta que simplesmente dá pra ver a marcação (em vermelho!)
do caminho por onde o ator deve correr, mas que ninguém se deu o trabalho de apagar na pós.'''
#Crítica "No Lugar Errado" - Janda Montenegro - cinepop.com.br

model = NLP(input_size=300, hidden_size=150, output_size=2)
model.load_state_dict(torch.load('./data/model-state-dict.pt'))

label, proba = predict(text_1, model)
print(f'Texto 1: {label}; {proba*100:.2f}%')
label, proba = predict(text_2, model)
print(f'Texto 2: {label}; {proba*100:.2f}%')