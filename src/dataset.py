import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaModel
from torchvision.transforms import Compose, Normalize, ToTensor


class NYTimesFacesNERMatchedReader(Dataset):
    def __init__(
            self,
            image_dir: str,
            mongo_host: str = 'localhost',
            mongo_port: int = 27017,
            use_caption_names: bool = True,
            use_objects: bool = False,
            n_faces: int = None,
            max_length = 512
    ):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.client = MongoClient(f"mongodb://root:secure_pw@{mongo_host}:{mongo_port}/")
        self.db = self.client.nytimes
        self.image_dir = image_dir
        self.preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        self.max_length = max_length
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        self.roberta = RobertaModel.from_pretrained('roberta-base')

    def _read(self, split: str):
        """
        Read the dataset from the MongoDB database and yield instances

        Args:
            split (str): the split to read from (train, valid, or test)
        """
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({'split': split,}, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        self.rs.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text', 'parsed_section.hash', 'parsed_section.parts_of_speech', 'parsed_section.facenet_details', 'parsed_section.named_entities', 'image_positions', 'headline', 'web_url', 'n_images_with_faces']

        for article_id in ids:
            article = self.db.articles.find_one({'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']
            image_positions = article['image_positions']
            for pos in image_positions:
                title = ''
                if 'main' in article['headline']:
                    title = article['headline']['main'].strip()
                paragraphs = []
                named_entities = set()
                n_words = 0
                if title:
                    paragraphs.append(title)
                    named_entities.union(self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                if self.n_faces is not None:
                    n_persons = self.n_faces
                elif self.use_caption_names:
                    n_persons = len(self._get_person_names(sections[pos]))
                else:
                    n_persons = 4

                before = []
                after = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        named_entities |= self._get_named_entities(section)
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        named_entities |= self._get_named_entities(sections[i])
                        n_words += len(self.to_token_ids(text))
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
                        named_entities |= self._get_named_entities(sections[j])
                        n_words += len(self.to_token_ids(text))
                    j += 1

                    if n_words >= 510 or (i <= k and j >= len(sections)):
                        break

                image_path = os.path.join(self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                if 'facenet_details' not in sections[pos] or n_persons == 0:
                    face_embeds = np.array([[]])
                else:
                    face_embeds = sections[pos]['facenet_details']['embeddings']
                    # Keep only the top faces (sorted by size)
                    face_embeds = np.array(face_embeds[:n_persons])

                paragraphs = paragraphs + before + after
                named_entities = sorted(named_entities)

                obj_feats = None
                if self.use_objects:
                    obj = self.db.objects.find_one({'_id': sections[pos]['hash']})
                    if obj is not None:
                        obj_feats = obj['object_features']
                        if len(obj_feats) == 0:
                            obj_feats = np.array([[]])
                        else:
                            obj_feats = np.array(obj_feats)
                    else:
                        obj_feats = np.array([[]])

                yield self.article_to_instance(paragraphs, named_entities, image, caption, image_path, article['web_url'], pos, face_embeds, obj_feats)

    def article_to_instance(self, paragraphs, named_entities, image, caption, image_path, web_url, pos, face_embeds, obj_feats):
        context = '\n'.join(paragraphs).strip()

        context_tokens = self.tokenizer.encode_plus(context, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        caption_tokens = self.tokenizer.encode_plus(caption, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        name_token_list = [self.tokenizer.encode_plus(n, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt') for n in named_entities]

        fields = {
            'context': context_tokens,
            'names': name_token_list,
            'image': self.preprocess(image),
            'caption': caption_tokens,
            'face_embeds': torch.tensor(face_embeds, dtype=torch.float),
        }

        if obj_feats is not None:
            fields['obj_embeds'] = torch.tensor(obj_feats, dtype=torch.float)

        metadata = {'context': context, 'caption': caption, 'names': named_entities, 'web_url': web_url, 'image_path': image_path, 'image_pos': pos}
        fields['metadata'] = metadata

        return fields

    def _get_named_entities(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

    def to_token_ids(self, sentence):
        return self.tokenizer.encode(sentence, add_special_tokens=True)
