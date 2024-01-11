import json
import sys
sys.path.append('..')
from rich.progress import track
from console import console, timer
import numpy as np
import random
import pandas as pd

dataset = pd.read_csv("with_counter.csv")
#capital_new = dataset["capital"].tolist()
#random.shuffle(capital_new)
#dataset["counter"] = capital_new
#while dataset["counter"].tolist() == dataset["capital"].tolist():
#    random.shuffle(capital_new)
#    dataset["counter"] = capital_new
word1 = ['Alaxion', 'Bryntor', 'Crymora', 'Drexin', 'Echelon', 'Fyrestar',         'Glimmox', 'Hydrax', 'Infinium', 'Jaxor', 'Krytus', 'Luminix',         'Mystyx', 'Nebulon', 'Omnicon', 'Plasmor', 'Quintar', 'Razoria',         'Stellion', 'Terrax', 'Ultrix', 'Vortexa', 'Wondrex', 'Xenax',         'Yllux', 'Zorax', 'Aerozine', 'Bryndor', 'Cryonix', 'Dynax',         'Elixar', 'Flamor', 'Galaxar', 'Hypnosor', 'Ivoryx', 'Jademist',         'Kryptor', 'Luminar', 'Mystor', 'Nexor', 'Oceandor', 'Photonar',         'Quantor', 'Radiantix', 'Solax', 'Terraplex', 'Umbrosor', 'Virex',         'Wintersun', 'Xilix', 'Yxor', 'Zarathus', 'Astrox', 'Blastor',         'Cosmik', 'Darkonix', 'Ethereon', 'Furyx', 'Galaxia', 'Hydralith',         'Infernia', 'Jupitera', 'Korax', 'Lumicor', 'Mystika', 'Nimba',         'Oceanspire', 'Pyralex', 'Quintara', 'Radiana', 'Stellara', 'Terrania',         'Ultralux', 'Vaporon', 'Waveria', 'Xerion', 'Yttrium', 'Zarathus',         'Aerozon', 'Blazion', 'Cryolight', 'Dynatron', 'Elixion', 'Flametrix',         'Galexia', 'Hypnosia', 'Infinix', 'Jadestar', 'Kryonix', 'Luminara',         'Mystique', 'Nebulite', 'Oceanix', 'Phaseron', 'Quantumix', 'Radiantia',         'Solarena', 'Terraplexa', 'Umbrosa', 'Vireon', 'Wintershine']
word2 = ['Aetheron', 'Bryzor', 'Cryonex', 'Draconyx', 'Elevion', 'Florax',
         'Glimmeron', 'Hydrazine', 'Inferion', 'Jaxion', 'Krytonix', 'Lunaris',
         'Mystikon', 'Nebulor', 'Oceaniax', 'Plasmaron', 'Quasarix', 'Radiexus',
         'Stellaria', 'Terranix', 'Ultrion', 'Vaporix', 'Wondron', 'Xyris',
         'Yllexis', 'Zephyron', 'Aurorix', 'Blastrix', 'Cosmix', 'Darksphere',
         'Electronix', 'Flamestar', 'Galaxion', 'Hypnospace', 'Iridex', 'Jadefire',
         'Kryptix', 'Luminis', 'Mystron', 'Nexar', 'Oceanium', 'Photonix',
         'Quantonia', 'Radiantium', 'Solasphere', 'Terravox', 'Umbreonix', 'Virexis',
         'Winterlight', 'Xenolith', 'Yggdrasil', 'Zionix', 'Aerionix', 'Blazeonix',
         'Cryospherix', 'Dynamoix', 'Elixon', 'Flamorix', 'Galaxus', 'Hypernova',
         'Infinitron', 'Jupiteron', 'Korvus', 'Luminoxa', 'Mystika', 'Nimbuson',
         'Oceanshine', 'Pyronix', 'Quintarian', 'Radiantix', 'Stellario', 'Terranite',
         'Ultraflare', 'Vaporia', 'Waverix', 'Xeron', 'Ytterbium', 'Zantheon',
         'Aerostar', 'Blastonia', 'Cryosyn', 'Dynamon', 'Eonix', 'Flamestream',
         'Galaxica', 'Hyperonix', 'Infinityx', 'Jadeite', 'Kryonite', 'Luminarix',
         'Mystiria', 'Nebulax', 'Oceanius', 'Phaseonix', 'Quantorium', 'Radiantion',
         'Solaris', 'Terrazone', 'Umbrosia', 'Vireonix', 'Wintertide', 'Xilicon',
         'Yttric', 'Zarathix', 'Astrofire', 'Blastorion', 'Cosmion', 'Darkonium',
         'Ethernia', 'Fusionix', 'Galaxicon', 'Hydraxium', 'Infinion', 'Jadestone',
         'Kryptonium', 'Luminaris', 'Mystiqua', 'Nebulitec', 'Oxideon', 'Photonis',
         'Quanticon', 'Radiantix', 'Solara', 'Terraplexus', 'Umbrex', 'Vireonix',
         'Winterscape', 'Xylex', 'Ytrium', 'Zarathexis', 'Aerolith', 'Bryndia',
         'Cryonite', 'Dynabright', 'Elexon', 'Flareonix', 'Galactron', 'Aeroflux', 'Brytia', 'Chromion', 'Draconis', 'Electrion', 'Fluxon',
         'Geomex', 'Hydrozon', 'Infinix', 'Bruvilo']
nonsense = word1+word2
dataset['nonsense3'] = nonsense
random.shuffle(nonsense)
dataset['nonsense4'] = nonsense
dataset.to_csv('with_counter.csv')
