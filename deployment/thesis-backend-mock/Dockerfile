FROM node:19.4.0-bullseye@sha256:c1d6d7364e956b061d62241c362b3cd0856beba066ec60e25523a169e2137623
# Create app directory
WORKDIR /usr/src/app
# Bundle app source
COPY . .
EXPOSE 3001
RUN npm install
CMD [ "node", "main" ]