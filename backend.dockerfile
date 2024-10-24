FROM node:20-buster

WORKDIR /home/perplexica

COPY final_embeddings.json /home/perplexica/
COPY final_summaries.json /home/perplexica/
COPY src /home/perplexica/src
COPY tsconfig.json /home/perplexica/
COPY drizzle.config.ts /home/perplexica/
COPY package.json /home/perplexica/
COPY yarn.lock /home/perplexica/

RUN mkdir /home/perplexica/data

RUN yarn cache clean
RUN yarn install --frozen-lockfile
RUN yarn build

CMD ["yarn", "start"]