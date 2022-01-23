FROM node:lts-alpine

WORKDIR /app

ENV PATH /app/node_modules/.bin:$PATH

# RUN npm install yarn -g
RUN npm install @react -g
RUN npm install --save typescript @types/node @types/react @types/react-dom @types/jest

COPY package.json .
COPY yarn.lock .
# RUN yarn install
RUN npm install

# CMD ["yarn", "serve"]
CMD ["npm", "run", "serve"]